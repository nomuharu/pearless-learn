# Task 2-5: 推論エンジン + Named Pipe スタブ実装（AC-017〜020）

## タスク概要

`DataSourceInterface` を継承したダミーデータ生成スタブ `NamedPipeStub` と、scaler.pkl ロード・feature_engineering 適用・モデル推論・時間計測を行う `InferenceEngine` を実装する。

## 対象ファイル

- `inference/pipe_stub.py` (新規)
- `inference/engine.py` (新規)

## 調査対象

- `/home/nomu/claude_code/pearless/inference/interface.py` (Task 0-2 の DataSourceInterface)
- `/home/nomu/claude_code/pearless/models/base.py` (Task 2-1 の BaseModel)
- `/home/nomu/claude_code/pearless/pipeline.py` (Task 1-1 の feature_engineering)
- `/home/nomu/claude_code/pearless/docs/design/fx-prediction-design.md` (§ InferenceEngine.predict() 戻り値、§ Contract Definitions)
- `/home/nomu/claude_code/pearless/tests/integration/test_inference_engine.int.py` (AC-017〜020 アサーション仕様)

## 実装手順

### Step 1: NamedPipeStub 実装（inference/pipe_stub.py）

```python
import numpy as np
import pandas as pd

from inference.interface import DataSourceInterface


class NamedPipeStub(DataSourceInterface):
    """DataSourceInterface の Named Pipe スタブ実装。

    MT4 Named Pipe 本番実装前のダミーデータ生成器。
    固定シードで再現性のあるランダム OHLCV データを生成する。
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed=seed)

    def fetch_latest_ohlcv(self, n_bars: int = 60) -> pd.DataFrame:
        """直近 n_bars 本の合成 OHLCV データを生成する。"""
        close = 150.0 + self._rng.normal(0, 0.5, n_bars).cumsum()
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n_bars, freq="5min"),
            "open": close + self._rng.uniform(-0.1, 0.1, n_bars),
            "high": close + self._rng.uniform(0.0, 0.3, n_bars),
            "low": close - self._rng.uniform(0.0, 0.3, n_bars),
            "close": close,
            "volume": self._rng.integers(100, 1000, n_bars).astype(float),
        })
```

### Step 2: InferenceEngine 実装（inference/engine.py）

```python
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from inference.interface import DataSourceInterface
from models.base import BaseModel
from pipeline import feature_engineering


_SIGNAL_MAP: dict[int, str] = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}


class InferenceEngine:
    """scaler.pkl ロード・特徴量エンジニアリング・モデル推論を統合する推論エンジン。

    戻り値仕様（predict()）:
        {
            "signal": "UP" | "DOWN" | "NEUTRAL",
            "probabilities": {"UP": float, "DOWN": float, "NEUTRAL": float},
            "inference_ms": float,
        }
    """

    def __init__(
        self,
        model: BaseModel,
        scaler_path: str | Path,
        data_source: DataSourceInterface,
    ) -> None:
        self._model = model
        self._data_source = data_source
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        self._model.eval()

    def predict(self) -> dict[str, object]:
        """推論を実行しシグナルと確率を返す。"""
        start = time.perf_counter()

        # データ取得
        df_raw = self._data_source.fetch_latest_ohlcv(n_bars=60)

        # feature_engineering 適用
        df_features = feature_engineering(df_raw)

        # 正規化
        x_raw = df_features.values.reshape(1, -1, BaseModel.N_FEATURES).astype(np.float32)
        x_scaled = self._scaler.transform(x_raw.reshape(-1, BaseModel.N_FEATURES))
        x_tensor = torch.tensor(
            x_scaled.reshape(1, BaseModel.SEQ_LEN, BaseModel.N_FEATURES),
            dtype=torch.float32,
        )

        # 推論
        with torch.no_grad():
            probs_tensor = self._model(x_tensor)  # (1, 3)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        probs = probs_tensor[0].tolist()
        signal_idx = int(np.argmax(probs))

        return {
            "signal": _SIGNAL_MAP[signal_idx],
            "probabilities": {
                "UP": probs[0],
                "DOWN": probs[1],
                "NEUTRAL": probs[2],
            },
            "inference_ms": elapsed_ms,
        }
```

## 品質保証メカニズム

| メカニズム | 確認内容 | 適用箇所 |
|---|---|---|
| スタブ連続 100 回推論エラーゼロ | 推論エンジン安定性 (AC-019) | Task 2-7 のテスト |
| パイプライン各ステップ後アサーション | feature_engineering の NaN チェック | `pipeline.feature_engineering()` 内 |

## 動作確認方法

```python
from inference.pipe_stub import NamedPipeStub
from inference.engine import InferenceEngine
from models.patchtst import PatchTST
import pickle
import numpy as np
from pathlib import Path
import tempfile

# scaler.pkl 作成（テスト用）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(np.random.randn(100, 16))
with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    pickle.dump(scaler, f)
    scaler_path = f.name

# 推論実行
model = PatchTST()
stub = NamedPipeStub()
engine = InferenceEngine(model=model, scaler_path=scaler_path, data_source=stub)
result = engine.predict()

# AC-018 確認
assert result["signal"] in {"UP", "DOWN", "NEUTRAL"}, f"Invalid signal: {result['signal']}"
assert set(result["probabilities"].keys()) == {"UP", "DOWN", "NEUTRAL"}
assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
assert isinstance(result["inference_ms"], float)
print(f"OK: signal={result['signal']}, inference_ms={result['inference_ms']:.1f}ms")
```

**成功基準**:
- `NamedPipeStub()` 経由で推論が成功し、シグナルと確率3値が返る (AC-018)
- 推論時間 100 回平均 50ms 未満 (AC-017)
- `DataSourceInterface` の差し替えがインターフェース変更なしで可能 (AC-020)

**検証レベル**: L1（機能動作検証）

## 完了条件

- [ ] Implementation: `NamedPipeStub()` 経由で推論が成功し、`{"signal", "probabilities", "inference_ms"}` が返る (AC-018)
- [ ] Quality: 推論時間 100 回平均 50ms 未満 (AC-017)
- [ ] Integration: `DataSourceInterface` のサブクラスを差し替えても `InferenceEngine` が同一インターフェースで動作する (AC-020)
