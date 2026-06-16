//+------------------------------------------------------------------+
//| PearlessDirection.mq4                                             |
//| 方向予測・成行エントリーEA（Pythonシグナル連携）                  |
//|                                                                    |
//| 構成:                                                              |
//|   1. 新しい5分足の確定時、直近バーをFiles\pearless_bars.csvに出力 |
//|      （PearlessBreakout.mq4 と同じファイル→Python側で共用可）    |
//|   2. Python側(scripts/mt4_signal_writer_dir_v2.py)が             |
//|      Files\pearless_dir_signal.csv に                             |
//|      「バー時刻,p_move,direction,p_up」を書き込む                 |
//|      direction: "BUY" / "SELL" / "SKIP"                           |
//|   3. EAはdirectionに従い成行エントリー                            |
//|   4. 次の5分足確定時（= 5分後）に成行決済                        |
//|      TP/SL なし、時間エグジット固定                               |
//|                                                                    |
//| 注意: 本ファイルはMT4環境がないため未コンパイル・未実走。          |
//|       MetaEditorでコンパイルし、まずデモ口座で検証すること。      |
//+------------------------------------------------------------------+
#property strict

input int    MagicNumber    = 20260616;
input double Leverage       = 25.0;    // 使用レバレッジ
input double MaxRiskPct     = 100.0;   // 有効証拠金の何%まで使うか
input int    SlippagePoints = 10;      // 許容スリッページ（point）
input int    ExportBars     = 300;     // Pythonに渡す直近バー数
input double MaxSpreadYen   = 0.005;   // 最大スプレッド（0.5銭超は見送り）
input bool   EnablePush     = true;    // スマホへのpush通知

datetime g_lastBarTime   = 0;     // 最後に処理した5分足の開始時刻
bool     g_hasPosition   = false; // 建玉保有中フラグ
int      g_lastDay       = -1;    // 日次サマリ通知用

//+------------------------------------------------------------------+
int OnInit()
{
   if(Period() != PERIOD_M5)
   {
      Print("M5チャートで動かしてください");
      return INIT_FAILED;
   }
   g_lastBarTime = Time[0];
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(Time[0] != g_lastBarTime)
   {
      g_lastBarTime = Time[0];
      OnNewBar();
   }
}

//+------------------------------------------------------------------+
void OnNewBar()
{
   // 0. 日付が変わっていたら前日の損益サマリを通知
   int today = TimeDay(Time[0]);
   if(g_lastDay != -1 && today != g_lastDay)
      NotifyDailySummary();
   g_lastDay = today;

   // 1. 前バーのポジションを成行決済（5分時間エグジット）
   if(g_hasPosition)
   {
      CloseAllPositions();
      g_hasPosition = false;
   }

   // 2. 直近バーをPython用にエクスポート
   ExportRecentBars();

   // 3. シグナルファイルを確認（Pythonの推論完了を最大3秒待つ）
   string direction = "";
   double pMove     = 0;
   double pUp       = 0;
   for(int retry = 0; retry < 30; retry++)
   {
      if(ReadSignal(Time[1], direction, pMove, pUp))
         break;
      Sleep(100);
   }

   if(direction == "" || direction == "SKIP")
      return;

   // 4. スプレッドチェック
   double spreadYen = Ask - Bid;
   if(spreadYen > MaxSpreadYen)
   {
      Print("シグナルあり(", direction, " p_move=", pMove, ")だがスプレッド拡大中のため見送り: ",
            DoubleToString(spreadYen, Digits));
      return;
   }

   // 5. 成行エントリー
   PlaceMarketOrder(direction, pMove, pUp);
}

//+------------------------------------------------------------------+
//| 直近バーのOHLCVをCSV出力（PearlessBreakout.mq4 と共用）          |
//+------------------------------------------------------------------+
void ExportRecentBars()
{
   int fh = FileOpen("pearless_bars.csv", FILE_WRITE|FILE_CSV, ',');
   if(fh == INVALID_HANDLE) { Print("bars出力失敗: ", GetLastError()); return; }
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
//| 形式: "yyyy.mm.dd hh:mi,p_move,direction,p_up"                   |
//+------------------------------------------------------------------+
bool ReadSignal(datetime barTime, string &direction, double &pMove, double &pUp)
{
   int fh = FileOpen("pearless_dir_signal.csv", FILE_READ|FILE_CSV, ',');
   if(fh == INVALID_HANDLE) return false;
   string ts   = FileReadString(fh);
   string pm   = FileReadString(fh);
   string dir  = FileReadString(fh);
   string pu   = FileReadString(fh);
   FileClose(fh);
   if(StrToTime(ts) != barTime) return false;  // 古いシグナルは無視
   pMove     = StrToDouble(pm);
   direction = dir;
   pUp       = StrToDouble(pu);
   return true;
}

//+------------------------------------------------------------------+
//| 有効証拠金・レバレッジからロットを自動計算                        |
//+------------------------------------------------------------------+
double CalcLotSize()
{
   double equity   = AccountEquity();
   double tradable = equity * (MaxRiskPct / 100.0) * Leverage;
   double lotRaw   = tradable / 100000.0;
   double lotStep  = MarketInfo(Symbol(), MODE_LOTSTEP);
   double lotMin   = MarketInfo(Symbol(), MODE_MINLOT);
   double lotMax   = MarketInfo(Symbol(), MODE_MAXLOT);
   double lot      = MathFloor(lotRaw / lotStep) * lotStep;
   lot = MathMax(lotMin, MathMin(lot, lotMax));
   return NormalizeDouble(lot, 2);
}

//+------------------------------------------------------------------+
//| 成行エントリー                                                    |
//+------------------------------------------------------------------+
void PlaceMarketOrder(string direction, double pMove, double pUp)
{
   double lotSize = CalcLotSize();
   int    opType;
   double price;
   string label;

   if(direction == "BUY")
   {
      opType = OP_BUY;
      price  = Ask;
      label  = "pearless-dir-buy";
   }
   else  // SELL
   {
      opType = OP_SELL;
      price  = Bid;
      label  = "pearless-dir-sell";
   }

   int ticket = OrderSend(Symbol(), opType, lotSize, price,
                          SlippagePoints, 0, 0,
                          label, MagicNumber, 0,
                          opType == OP_BUY ? clrBlue : clrRed);
   if(ticket < 0)
   {
      Print("OrderSend失敗: err=", GetLastError(),
            " direction=", direction,
            " price=", DoubleToString(price, Digits));
      return;
   }

   g_hasPosition = true;
   Print("エントリー: ", direction, " lot=", DoubleToString(lotSize, 2),
         " @", DoubleToString(price, Digits),
         " p_move=", DoubleToString(pMove, 4),
         " p_up=", DoubleToString(pUp, 4));

   if(EnablePush)
      SendNotification(StringConcatenate(
         "[pearless-dir] ", direction, " ", DoubleToString(lotSize, 2), "lot @",
         DoubleToString(price, Digits),
         " (", TimeToString(TimeCurrent(), TIME_MINUTES), ")"));
}

//+------------------------------------------------------------------+
//| 全建玉を成行決済（5分時間エグジット）                             |
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
//| 前日の確定損益を集計してpush通知                                  |
//+------------------------------------------------------------------+
void NotifyDailySummary()
{
   datetime dayStart = StrToTime(TimeToString(Time[1], TIME_DATE));
   double pnl = 0;
   int    n   = 0;
   for(int i = OrdersHistoryTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY)) continue;
      if(OrderMagicNumber() != MagicNumber || OrderSymbol() != Symbol()) continue;
      if(OrderType() != OP_BUY && OrderType() != OP_SELL) continue;
      if(OrderCloseTime() < dayStart) break;
      pnl += OrderProfit() + OrderSwap() + OrderCommission();
      n++;
   }
   if(n == 0) return;
   if(EnablePush)
      SendNotification(StringConcatenate(
         "[pearless-dir] 日次サマリ ", TimeToString(dayStart, TIME_DATE),
         ": ", IntegerToString(n), "トレード 損益 ",
         DoubleToString(pnl, 0), AccountCurrency()));
   Print("日次サマリ: ", n, "トレード 損益 ", pnl);
}
//+------------------------------------------------------------------+
