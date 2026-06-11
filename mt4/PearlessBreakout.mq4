//+------------------------------------------------------------------+
//| PearlessBreakout.mq4                                              |
//| OCOブレイクアウト・エントリーEA（Pythonシグナル連携）            |
//|                                                                    |
//| 構成:                                                              |
//|   1. 新しい5分足の確定時、直近バーをFiles\pearless_bars.csvに出力 |
//|   2. Python側(scripts/mt4_signal_writer.py)がそれを読み推論し、   |
//|      Files\pearless_signal.csv に「バー時刻,p_move」を書き込む    |
//|   3. EAは自分が出力したバー時刻と一致するシグナルを見つけたら、   |
//|      確定値P0の上下±DeltaPointsにOCO逆指値を設置                  |
//|   4. 片方が約定したらもう片方を削除（OCOエミュレーション）        |
//|   5. バー終了時にポジションを成行決済、未約定の逆指値は削除       |
//|                                                                    |
//| 注意: 本ファイルはMT4環境がないため未コンパイル・未実走。          |
//|       MetaEditorでコンパイルし、まずデモ口座/Strategy Testerで     |
//|       検証すること。                                               |
//+------------------------------------------------------------------+
#property strict

input int    MagicNumber   = 20260611;
input double LotSize       = 0.01;     // ロット
input double MinPMove      = 0.55;     // p_move の発注閾値
input double DeltaYen      = 0.015;    // 逆指値の距離（円。0.015 = 1.5銭）
input int    SlippagePoints = 10;      // 許容スリッページ（point）
input int    ExportBars    = 300;      // Pythonに渡す直近バー数
input double MaxSpreadYen  = 0.005;    // この値より広いスプレッド時は発注しない（0.5銭）
                                       // 高p_moveは指標発表と重なりやすく、その瞬間は
                                       // スプレッドが数銭に拡大してエッジが消えるため必須

datetime g_lastBarTime = 0;        // 最後に処理した5分足の開始時刻
datetime g_signalBarTime = 0;      // OCOを設置したシグナルのバー時刻
bool     g_ocoActive = false;

//+------------------------------------------------------------------+
int OnInit()
{
   if(Period() != PERIOD_M5)
   {
      Print("M5チャートで動かしてください");
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick()
{
   // --- 新しいバーの検出（= 直前バーの確定） ---
   if(Time[0] != g_lastBarTime)
   {
      g_lastBarTime = Time[0];
      OnNewBar();
   }

   // --- OCO管理: 片方約定でもう片方を削除 ---
   if(g_ocoActive)
      ManageOco();
}

//+------------------------------------------------------------------+
void OnNewBar()
{
   // 1. 前バーまでの建玉と注文を清算（1本ホールドの決済）
   CloseAllPositions();
   DeleteAllPendings();
   g_ocoActive = false;

   // 2. 直近バーをPython用にエクスポート
   ExportRecentBars();

   // 3. シグナルファイルを確認（Pythonの推論完了を最大3秒待つ）
   double pMove = 0;
   for(int retry = 0; retry < 30; retry++)
   {
      if(ReadSignal(Time[1], pMove))
         break;
      Sleep(100);
   }

   // 4. 閾値を超えていて、かつスプレッドが正常範囲ならOCO設置
   double spreadYen = Ask - Bid;
   if(pMove >= MinPMove && spreadYen <= MaxSpreadYen)
      PlaceOco(Close[1]);
   else if(pMove >= MinPMove)
      Print("シグナルあり(p_move=", pMove, ")だがスプレッド拡大中のため見送り: ",
            DoubleToString(spreadYen, Digits));
}

//+------------------------------------------------------------------+
//| 直近バーのOHLCVをCSV出力（Python側の特徴量計算用）                |
//+------------------------------------------------------------------+
void ExportRecentBars()
{
   int fh = FileOpen("pearless_bars.csv", FILE_WRITE|FILE_CSV, ',');
   if(fh == INVALID_HANDLE) { Print("bars出力失敗: ", GetLastError()); return; }
   // 古い順に出力。バーは確定済みのもののみ（index 1 から）
   for(int i = ExportBars; i >= 1; i--)
   {
      FileWrite(fh,
                TimeToString(Time[i], TIME_DATE|TIME_MINUTES),
                DoubleToString(Open[i], Digits),
                DoubleToString(High[i], Digits),
                DoubleToString(Low[i], Digits),
                DoubleToString(Close[i], Digits),
                DoubleToString(Volume[i], 0));
   }
   FileClose(fh);
}

//+------------------------------------------------------------------+
//| シグナルファイル読み込み。barTimeが一致すればtrue                 |
//| 形式: "yyyy.mm.dd hh:mi,0.62"（Python側が書く）                   |
//+------------------------------------------------------------------+
bool ReadSignal(datetime barTime, double &pMove)
{
   int fh = FileOpen("pearless_signal.csv", FILE_READ|FILE_CSV, ',');
   if(fh == INVALID_HANDLE) return false;
   string ts = FileReadString(fh);
   string pm = FileReadString(fh);
   FileClose(fh);
   if(StrToTime(ts) != barTime) return false;  // 古いシグナルは無視
   pMove = StrToDouble(pm);
   return true;
}

//+------------------------------------------------------------------+
//| OCO逆指値の設置（買い: P0+δ / 売り: P0−δ、有効期限はバー終了）   |
//+------------------------------------------------------------------+
void PlaceOco(double p0)
{
   datetime expiry = Time[0] + PeriodSeconds();  // このバーの終わり
   double buyTrig  = NormalizeDouble(p0 + DeltaYen, Digits);
   double sellTrig = NormalizeDouble(p0 - DeltaYen, Digits);

   int buyTicket = OrderSend(Symbol(), OP_BUYSTOP, LotSize, buyTrig,
                             SlippagePoints, 0, 0,
                             "pearless-oco", MagicNumber, expiry, clrBlue);
   if(buyTicket < 0) Print("BuyStop失敗: ", GetLastError());

   int sellTicket = OrderSend(Symbol(), OP_SELLSTOP, LotSize, sellTrig,
                              SlippagePoints, 0, 0,
                              "pearless-oco", MagicNumber, expiry, clrRed);
   if(sellTicket < 0) Print("SellStop失敗: ", GetLastError());

   g_ocoActive = (buyTicket >= 0 || sellTicket >= 0);
   g_signalBarTime = Time[0];
}

//+------------------------------------------------------------------+
//| 片方が約定していたら残りの逆指値を削除                            |
//+------------------------------------------------------------------+
void ManageOco()
{
   bool hasPosition = false;
   for(int i = 0; i < OrdersTotal(); i++)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderMagicNumber() != MagicNumber || OrderSymbol() != Symbol()) continue;
      if(OrderType() == OP_BUY || OrderType() == OP_SELL)
         hasPosition = true;
   }
   if(!hasPosition) return;

   DeleteAllPendings();
   g_ocoActive = false;  // 以降はバー終了の決済待ちのみ
}

//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderMagicNumber() != MagicNumber || OrderSymbol() != Symbol()) continue;
      if(OrderType() == OP_BUY)
         OrderClose(OrderTicket(), OrderLots(), Bid, SlippagePoints, clrGray);
      else if(OrderType() == OP_SELL)
         OrderClose(OrderTicket(), OrderLots(), Ask, SlippagePoints, clrGray);
   }
}

//+------------------------------------------------------------------+
void DeleteAllPendings()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderMagicNumber() != MagicNumber || OrderSymbol() != Symbol()) continue;
      if(OrderType() == OP_BUYSTOP || OrderType() == OP_SELLSTOP)
         OrderDelete(OrderTicket());
   }
}
//+------------------------------------------------------------------+
