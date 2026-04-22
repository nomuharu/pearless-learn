"""pytest 設定: ルートディレクトリを sys.path に追加してモジュールインポートを解決する。"""
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent))
