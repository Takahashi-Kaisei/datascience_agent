# Data Science Agent

データサイエンスプロジェクト用のテンプレートリポジトリ。

## 環境構築

```bash
uv sync
```

## 実行

```bash
uv run python main.py
uv run jupyter notebook
```

## 開発

### Linting & Formatting
```bash
uv run ruff check
uv run ruff check --fix
uv run ruff format
```

### Type Checking
```bash
uv run mypy .
```

### Testing
```bash
uv run pytest
uv run pytest -v
uv run pytest --cov
```

## プロジェクト構成

```
.
├── src/datascience_agent/  # ソースコード
├── tests/                   # テストコード
├── data/
│   ├── raw/                 # 生データ（追跡対象外）
│   ├── processed/           # 処理済みデータ
│   └── external/            # 外部データ
├── notebooks/               # Jupyter Notebooks
└── outputs/
    ├── models/              # 学習済みモデル
    ├── reports/             # レポート
    └── figures/             # 図表
```