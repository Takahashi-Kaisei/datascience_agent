"""
YouTubeチャットモデレータモデル用特徴量エンジニアリング

処理内容:
1. 独立メッセージ特徴量の抽出（時系列要素なし）
2. 各メッセージの内容からのみ特徴量を計算
3. ユーザーIDは特徴量に含めない（汎化性能のため）

特徴量カテゴリ:
- 基本統計（文字数、単語数等）
- 文字種別（ひらがな、カタカナ、漢字、アルファベット等の比率）
- パターン（連続文字、記号等）
- 内容フラグ（URL、メンション等）

Complexity: O(n * m) where n = messages, m = avg message length
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MessageFeatureExtractor:
    """
    メッセージ内容から独立した特徴量を抽出するクラス

    時系列要素を含まず、各メッセージのみから特徴量を計算
    """

    def __init__(self):
        # 文字パターンの正規表現
        self.hiragana_pattern = re.compile(r"[\u3040-\u309F]")
        self.katakana_pattern = re.compile(r"[\u30A0-\u30FF]")
        self.kanji_pattern = re.compile(r"[\u4E00-\u9FFF]")
        self.alphabet_pattern = re.compile(r"[a-zA-Z]")
        self.number_pattern = re.compile(r"[0-9]")
        self.special_pattern = re.compile(r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFFa-zA-Z0-9\s]")
        self.repeated_pattern = re.compile(r"(.)\1{2,}")  # 同じ文字が3回以上連続

    def extract_features(self, message: str) -> dict[str, float]:
        """
        単一メッセージから特徴量を抽出

        Args:
            message: メッセージ本文

        Returns:
            特徴量の辞書

        Complexity: O(m) where m = len(message)
        """
        # 非文字列や欠損値は空メッセージとして扱う
        if not isinstance(message, str) or pd.isna(message) or len(message) == 0:
            return self._get_empty_features()

        # 基本統計
        char_count = len(message)
        words = message.split()
        word_count = len(words)

        features = {
            # 基本統計
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0,
            # 文字種別（比率）
            "hiragana_ratio": len(self.hiragana_pattern.findall(message)) / char_count,
            "katakana_ratio": len(self.katakana_pattern.findall(message)) / char_count,
            "kanji_ratio": len(self.kanji_pattern.findall(message)) / char_count,
            "alphabet_ratio": len(self.alphabet_pattern.findall(message)) / char_count,
            "number_ratio": len(self.number_pattern.findall(message)) / char_count,
            "special_char_ratio": len(self.special_pattern.findall(message)) / char_count,
            # パターン
            "upper_case_ratio": sum(1 for c in message if c.isupper()) / char_count,
            "repeated_char_ratio": self._calc_repeated_ratio(message),
            "exclamation_count": message.count("!") + message.count("！"),
            "question_count": message.count("?") + message.count("？"),
            # 内容フラグ
            "contains_url": 1.0 if ("http" in message or "www." in message) else 0.0,
            "contains_mention": 1.0 if ("@" in message or "＠" in message) else 0.0,
            "contains_hashtag": 1.0 if ("#" in message or "＃" in message) else 0.0,
            # テキスト統計
            "unique_char_ratio": len(set(message)) / char_count,
            "line_break_count": message.count("\n"),
        }

        return features

    def _get_empty_features(self) -> dict[str, float]:
        """空メッセージ用のデフォルト特徴量"""
        return {
            "char_count": 0.0,
            "word_count": 0.0,
            "avg_word_length": 0.0,
            "hiragana_ratio": 0.0,
            "katakana_ratio": 0.0,
            "kanji_ratio": 0.0,
            "alphabet_ratio": 0.0,
            "number_ratio": 0.0,
            "special_char_ratio": 0.0,
            "upper_case_ratio": 0.0,
            "repeated_char_ratio": 0.0,
            "exclamation_count": 0.0,
            "question_count": 0.0,
            "contains_url": 0.0,
            "contains_mention": 0.0,
            "contains_hashtag": 0.0,
            "unique_char_ratio": 0.0,
            "line_break_count": 0.0,
        }

    def _calc_repeated_ratio(self, message: str) -> float:
        """
        連続した同じ文字の比率を計算

        Args:
            message: メッセージ本文

        Returns:
            連続文字の比率（0.0 ~ 1.0）
        """
        if len(message) < 3:
            return 0.0

        matches = self.repeated_pattern.finditer(message)
        repeated_chars = sum(len(match.group(0)) for match in matches)
        return repeated_chars / len(message)

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame全体の特徴量を抽出

        Args:
            df: 入力データ（bodyカラムが必要）

        Returns:
            特徴量を追加したDataFrame

        Complexity: O(n * m) where n = len(df), m = avg message length
        """
        print(f"特徴量抽出中... {len(df):,}件")

        features_list = []
        for idx, row in df.iterrows():
            features = self.extract_features(row["body"])
            features["index"] = idx
            features_list.append(features)

            if (idx + 1) % 1000 == 0:
                print(f"  処理済み: {idx + 1:,}/{len(df):,}")

        features_df = pd.DataFrame(features_list)
        features_df.set_index("index", inplace=True)

        # 元のDataFrameと結合
        result = pd.concat([df, features_df], axis=1)

        print(f"特徴量抽出完了: {len(features_df.columns)}特徴量")
        return result


def normalize_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    特徴量を正規化（Trainでfit、全セットでtransform）

    Args:
        train_df: 訓練データ
        val_df: 検証データ
        test_df: テストデータ
        feature_cols: 正規化する特徴量カラム

    Returns:
        (train_normalized, val_normalized, test_normalized, scaler)
    """
    print("\n特徴量正規化中...")

    scaler = StandardScaler()

    # Trainでfit
    train_features = train_df[feature_cols].values
    scaler.fit(train_features)

    # 全セットでtransform
    train_df_norm = train_df.copy()
    val_df_norm = val_df.copy()
    test_df_norm = test_df.copy()

    train_df_norm[feature_cols] = scaler.transform(train_features)
    val_df_norm[feature_cols] = scaler.transform(val_df[feature_cols].values)
    test_df_norm[feature_cols] = scaler.transform(test_df[feature_cols].values)

    print(f"正規化完了: {len(feature_cols)}特徴量")
    return train_df_norm, val_df_norm, test_df_norm, scaler


def save_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    output_dir: Path,
) -> None:
    """
    特徴量とラベルを保存（PyTorch用に最適化）

    Args:
        train_df: 訓練データ
        val_df: 検証データ
        test_df: テストデータ
        feature_cols: 特徴量カラム
        scaler: 正規化用スケーラー
        output_dir: 出力ディレクトリ
    """
    print("\n特徴量保存中...")

    # 特徴量とラベルを分離
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["is_anomaly"].values.astype(np.int64)

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["is_anomaly"].values.astype(np.int64)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["is_anomaly"].values.astype(np.int64)

    # NumPy形式で保存（メモリマップ対応）
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    # 特徴量名を保存
    np.save(output_dir / "feature_names.npy", np.array(feature_cols))

    # スケーラーを保存
    import joblib

    joblib.dump(scaler, output_dir / "scaler.joblib")

    # メタデータ
    metadata = {
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }

    import json

    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("保存完了:")
    print(f"  X_train.npy: {X_train.shape}")
    print(f"  X_val.npy: {X_val.shape}")
    print(f"  X_test.npy: {X_test.shape}")
    print(f"  特徴量名: {feature_cols}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("YouTubeチャットモデレータモデル - 特徴量エンジニアリング")
    print("=" * 60)

    INPUT_DIR = Path("outputs/features")
    OUTPUT_DIR = Path("outputs/features")

    # 1. データ読み込み
    print("\nデータ読み込み中...")
    train_df = pd.read_parquet(INPUT_DIR / "train_raw.parquet")
    val_df = pd.read_parquet(INPUT_DIR / "val_raw.parquet")
    test_df = pd.read_parquet(INPUT_DIR / "test_raw.parquet")

    print(f"Train: {len(train_df):,}件")
    print(f"Val: {len(val_df):,}件")
    print(f"Test: {len(test_df):,}件")

    # 2. 特徴量抽出
    extractor = MessageFeatureExtractor()

    train_features = extractor.transform_dataframe(train_df)
    val_features = extractor.transform_dataframe(val_df)
    test_features = extractor.transform_dataframe(test_df)

    # 3. 特徴量カラムの特定
    feature_cols = [
        col
        for col in train_features.columns
        if col not in ["authorName", "body", "label", "month", "is_anomaly"]
    ]

    print(f"\n抽出された特徴量: {len(feature_cols)}個")
    print(feature_cols)

    # 4. 正規化
    train_norm, val_norm, test_norm, scaler = normalize_features(
        train_features, val_features, test_features, feature_cols
    )

    # 5. 保存
    save_features(train_norm, val_norm, test_norm, feature_cols, scaler, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("特徴量エンジニアリング完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
