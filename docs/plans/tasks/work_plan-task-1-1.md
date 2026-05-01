# Task 1-1: pipeline.py 実装（feature_engineering / create_label / split / normalize）

## タスク概要

全モデルの学習データを生成する基盤パイプラインを実装する。16特徴量計算・3クラスラベル生成・時系列分割・正規化・numpy保存の5段階から構成され、各ステップ後にアサーションを設ける。SHA-256 冪等性を保証する。

## 対象ファイル

- `pipeline.py` (新規)
- `data/` ディレクトリ（出力先、.gitignore で除外済み）

## 調査対象

- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ Existing Codebase Analysis、§ Contract Definitions)
- `/home/nomu/claude_code/pearless/docs/plans/work_plan.md` (§ Task 1-1、§ Verification Strategy)
- `/home/nomu/claude_code/pearless/tests/integration/test_pipeline.int.py` (AC 定義・アサーション仕様確認)
- `/home/nomu/claude_code/pearless/docs/adr/ADR-0002-technical-indicator-library.md` (pandas-ta API マッピング)

## 実装手順

### Step 1: 16特徴量の定義

設計書の ADR-0002 に従い pandas-ta で以下の16特徴量を計算する:

| # | 特徴量 | pandas-ta API |
|---|---|---|
| 1 | open | raw |
| 2 | high | raw |
| 3 | low | raw |
| 4 | close | raw |
| 5 | volume | raw |
| 6 | RSI(14) | `ta.rsi(close, 14)` |
| 7 | MACD_line | `ta.macd(close).iloc[:, 0]` |
| 8 | MACD_signal | `ta.macd(close).iloc[:, 1]` |
| 9 | MACD_hist | `ta.macd(close).iloc[:, 2]` |
| 10 | ATR(14) | `ta.atr(high, low, close, 14)` |
| 11 | BB_upper | `ta.bbands(close).iloc[:, 2]` |
| 12 | BB_mid | `ta.bbands(close).iloc[:, 1]` |
| 13 | BB_lower | `ta.bbands(close).iloc[:, 0]` |
| 14 | SMA(20) | `ta.sma(close, 20)` |
| 15 | SMA(60) | `ta.sma(close, 60)` |
| 16 | EMA(9) | `ta.ema(close, 9)` |

実際の列数と列名は ADR-0002 を必ず確認してから実装する。

### Step 2: feature_engineering 実装

```python
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame から 16 特徴量を計算する。

    Args:
        df: datetime, open, high, low, close, volume を含む DataFrame。

    Returns:
        16 列の特徴量 DataFrame（先頭 NaN 行はドロップ済み）。

    Asserts:
        - shape[1] == 16
        - isna().sum().sum() == 0
    """
```

- 各指標計算後に結合
- `df.dropna()` で先頭NaN行をドロップ
- アサーション: `assert df_out.shape[1] == 16`、`assert df_out.isna().sum().sum() == 0`

### Step 3: create_label 実装

```python
def create_label(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float | None = None,
) -> np.ndarray:
    """3クラスラベルを生成する。UP=0, DOWN=1, NEUTRAL=2。

    threshold=None の場合: θ = diff.abs().quantile(0.75)
    """
```

- `diff = df["close"].shift(-horizon) - df["close"]`
- threshold が None の場合: `theta = diff.abs().quantile(0.75)`
- `UP(0): diff > theta`、`DOWN(1): diff < -theta`、`NEUTRAL(2): それ以外`
- アサーション: `set(np.unique(labels)) <= {0, 1, 2}`

### Step 4: create_windows 実装

```python
def create_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """時系列ウィンドウを生成する。

    Returns:
        X: shape (N, window_size, n_features), float32
        y: shape (N,), int64
    """
```

### Step 5: split_time_series 実装

```python
def split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    ratios: list[float] = [0.70, 0.15, 0.15],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """時系列順序を保ったまま train/val/test に分割する。シャッフル禁止。"""
```

- `n = len(X)` → `n_train = int(n * ratios[0])` → 整数切り捨て
- インデックスで順序分割（シャッフルなし）

### Step 6: normalize 実装

```python
def normalize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """StandardScaler を train のみで fit し、val/test には transform のみ適用。scaler.pkl を保存。"""
```

- `reshape(-1, n_features)` で 2D に変換して fit
- `scaler.pkl` を pickle で保存

### Step 7: run_pipeline 実装

```python
def run_pipeline(
    csv_path: str,
    output_dir: str,
    window_size: int = 60,
    horizon: int = 1,
    threshold: float | None = None,
    ratios: list[float] = [0.70, 0.15, 0.15],
) -> None:
    """end-to-end パイプライン実行。SHA-256 冪等性確認付き。"""
```

- SHA-256: `hashlib.sha256(pd.read_csv(csv_path).to_csv().encode()).hexdigest()` で入力ハッシュ記録
- 出力: `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`, `scaler.pkl`

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| パイプライン冪等性（SHA-256 検証）| 同一 CSV 入力 → 同一 numpy 出力 | `run_pipeline()` 内アサーション |
| パイプライン各ステップ後アサーション | shape・NaN 数・クラス分布 | 各関数内アサーション |

## 動作確認方法（Early Verification Point）

```python
# Early Verification Point: feature_engineering の出力確認
import pandas as pd
import numpy as np

# 合成データ準備
rng = np.random.default_rng(seed=42)
n = 300
close = 150.0 + rng.normal(0, 0.5, n).cumsum()
df = pd.DataFrame({
    "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
    "open": close + rng.uniform(-0.1, 0.1, n),
    "high": close + rng.uniform(0.0, 0.3, n),
    "low": close - rng.uniform(0.0, 0.3, n),
    "close": close,
    "volume": rng.integers(100, 1000, n).astype(float),
})

from pipeline import feature_engineering
df_features = feature_engineering(df)

# Early Verification Point
assert df_features.shape[1] == 16 and df_features.isna().sum().sum() == 0
print(f"OK: shape={df_features.shape}, NaN={df_features.isna().sum().sum()}")
```

```bash
# run_pipeline 実行確認
uv run python pipeline.py --csv-path data/USDJPY_M5.csv --output-dir data/
ls data/
# 期待: X_train.npy y_train.npy X_val.npy y_val.npy X_test.npy y_test.npy scaler.pkl
```

**成功基準**:
- `feature_engineering()` の出力が shape `(N, 16)` かつ NaN ゼロ（Early Verification Point）
- `uv run python pipeline.py` が正常終了し `data/` に 7 ファイルが生成される

**検証レベル**: L1（機能動作検証）

## 完了条件

- [x] Implementation: `uv run python pipeline.py` が正常終了し `data/` に X_train.npy 等 7 ファイルが生成される
- [x] Quality: 各ステップ後アサーション（shape・NaN・クラス分布）がパスする
- [x] Integration: `feature_engineering()` の出力が shape `(N, 16)` かつ NaN ゼロ（Early Verification Point）
