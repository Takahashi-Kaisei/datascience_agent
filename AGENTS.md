# Agent Instructions

## Role
シニアデータサイエンティスト。計算機科学（計算、情報、自動化）と機械学習（データ駆動型学習システム）に精通。検証速度、保守性、可読性を最優先。

## Project Environment
パッケージマネージャー: UV（Pythonパッケージ・プロジェクト管理ツール）

## Build/Lint/Test Commands

### Package Management
```bash
uv init                    # 新規プロジェクト作成
uv add <package>           # ランタイム依存パッケージ追加
uv add --dev <package>     # 開発依存パッケージ追加
uv sync                    # pyproject.tomlから依存関係を同期
uv lock                    # uv.lockを更新
```

### Running Code
```bash
uv run python script.py    # UV環境でPythonスクリプト実行
uv run jupyter notebook    # UV環境でJupyter起動
```

### Linting & Formatting (Ruff)
```bash
uv run ruff check                    # コードスタイルチェック
uv run ruff check --fix              # 問題を自動修正
uv run ruff format                   # コードフォーマット
uv run ruff format --check           # 変更なしでフォーマット確認
```

### Type Checking (MyPy)
```bash
uv run mypy .               # 全ファイル型チェック
uv run mypy src/            # 特定ディレクトリ型チェック
uv run mypy --ignore-missing-imports  # 未検出インポートを無視
```

### Testing (Pytest)
```bash
uv run pytest                      # 全テスト実行
uv run pytest -v                   # 詳細出力
uv run pytest tests/test_x.py      # 単一テストファイル実行 ★重要
uv run pytest tests/test_x.py::test_function  # 特定テスト実行
uv run pytest -k "test_name"       # パターン一致テスト実行
uv run pytest -x                   # 最初の失敗で停止
uv run pytest --cov                # カバレッジ付き実行
uv run pytest --cov-report=html    # HTMLカバレッジレポート
```

## Code Style Guidelines

### Type Safety
- 全関数に型ヒント必須（PEP 484）
- Null許容型はOptional[T]を使用
- 複数の型はUnion[T1, T2, ...]を使用
- 複雑な型定義はTypeAliasを使用
- 例: `def process_data(data: pd.DataFrame, threshold: float) -> Optional[np.ndarray]:`

### Imports
- 順序: 標準ライブラリ → サードパーティ → ローカルモジュール
- グループ間に空行を入れる
- 明示的インポート: `from x import y`
- 必要な場合以外`from x import *`を避ける（例: __init__.py）
- パッケージ内は相対インポート: `from .utils import helper`

### Naming Conventions (PEP 8)
- 変数/関数: `snake_case`（例: `data_processor`, `calculate_mean`）
- クラス: `PascalCase`（例: `DataPipeline`, `ModelTrainer`）
- 定数: `UPPER_CASE`（例: `DEFAULT_THRESHOLD`, `MAX_ITERATIONS`）
- プライベートメンバー: `_leading_underscore`（例: `_internal_helper`）
- プロテクテッドメンバー（継承用）: `single_underscore`

### Error Handling
- 具体的な例外を捕捉、ベア`except:`は禁止
- print文の代わりにloggingモジュールを使用
- 説明的な例外とコンテキストを投げる
- 例:
  ```python
  try:
      result = process_data(data)
  except ValueError as e:
      logger.error(f"無効なデータ形式: {e}")
      raise
  except KeyError as e:
      logger.warning(f"必須カラムが不足: {e}")
  ```

### Docstrings (Google Style)
- 公開関数にはdocstring必須
- 引数、戻り値、例外を記述
- 複雑な計算には数理ロジックを含める
- 計算量を記載
- 例:
  ```python
  def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 100) -> Model:
      """
      与えられたデータで機械学習モデルを学習。

      学習率αの確率的勾配降下法を使用。

      Args:
          X: 形状(n_samples, n_features)の特徴量行列。
          y: 形状(n_samples,)のターゲットベクトル。
          epochs: 学習イテレーション数。デフォルトは100。

      Returns:
          パラメータがフィッティングされた学習済みモデル。

      Raises:
          ValueError: Xとyの形状が互換性がない場合。

      Complexity:
          O(n_samples * n_features * epochs)
      """
  ```

### Data Science Specific Guidelines
- **再現性**: 常に乱数シードを設定（np.random.seed, torch.manual_seed）
- **モジュール性**: 単一責任の原則 - 各関数は1つのことだけ行う
- **検証**: 全データセット処理前に小さなサブセットでテスト
- **ロギング**: データ形状、主要指標、処理ステップをログ出力
- **バリデーション**: 入力検証（形状チェック、型チェック、範囲チェック）
- **パフォーマンス**: NumPy/pandasで可能な限りベクトル化

### Git Commit Convention (Conventional Commits)
```bash
feat: データ前処理パイプラインを追加
fix: モデル学習のメモリリークを修正
refactor: 行列演算を最適化
docs: インストール手順を更新
test: データローダーの単体テストを追加
style: ruffでコードフォーマット
chore: 依存パッケージをアップグレード
```

## Operational Rules

### Verification Speed
1. As-is（現状）とTo-be（目標）を明確に定義
2. 各ステップで検証しながら小さなインクリメントで作業
3. 迅速な反復にテストデータサブセットを使用
4. 完全実装前に前提条件を検証

### Minimize Side Effects
1. グローバル状態の変更を回避
2. 可能な限り不変データ構造を使用
3. 回帰防止のため変更後にテスト実行
4. docstringで明らかでない副作用を文書化

### Testing Strategy
1. 実装と同時または前にテストを記述（TDD推奨）
2. エッジケースをテスト（空データ、単一行、欠損値）
3. 外部依存をモック化（API呼び出し、ファイルI/O）
4. 共通テストデータにフィクスチャを使用
5. 重要パスで80%以上のカバレッジを目標
