"""
TF-IDF特徴量抽出

処理内容:
1. メッセージ本文からTF-IDFベクトルを生成
2. 日本語テキスト対応（文字N-gramベース）
3. Trainでfit、全セットでtransform

Complexity: O(n * v) where n = messages, v = vocabulary size
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def char_ngram_tokenizer(text, n_range=(1, 2)):
    """文字N-gramを生成（モジュールレベルで定義）"""
    tokens = []
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(text) - n + 1):
            tokens.append(text[i : i + n])
    return tokens


class CharNgramTokenizer:
    """Pickle可能なTokenizerクラス"""

    def __init__(self, n_range=(1, 2)):
        self.n_range = n_range

    def __call__(self, text):
        return char_ngram_tokenizer(text, self.n_range)


def create_tfidf_features(
    train_texts: list,
    val_texts: list,
    test_texts: list,
    max_features: int = 500,
    ngram_range: tuple = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
) -> tuple:
    """
    TF-IDF特徴量を生成

    日本語テキストの前処理は文字N-gramベースで実装
    （形態素解析は使用せず、シンプルかつ堅牢）

    Args:
        train_texts: 訓練データのテキストリスト
        val_texts: 検証データのテキストリスト
        test_texts: テストデータのテキストリスト
        max_features: 最大特徴量数
        ngram_range: N-gram範囲（(1, 2)はユニグラム+バイグラム）
        min_df: 最小文書頻度
        max_df: 最大文書頻度（比率）

    Returns:
        (X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer)
    """
    print(f"\nTF-IDF特徴量抽出（max_features={max_features}, ngram_range={ngram_range}）")

    # Pickle可能なtokenizer
    tokenizer = CharNgramTokenizer(n_range=ngram_range)

    # TfidfVectorizerの初期化
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,  # 日本語では小文字化不要
    )

    print("Vectorizer fitting on train data...")
    # Trainでfit
    X_train_tfidf = vectorizer.fit_transform(train_texts)

    print("Transforming all datasets...")
    # 全セットでtransform
    X_val_tfidf = vectorizer.transform(val_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    print(f"TF-IDF特徴量生成完了:")
    print(f"  Train: {X_train_tfidf.shape}")
    print(f"  Val: {X_val_tfidf.shape}")
    print(f"  Test: {X_test_tfidf.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def save_tfidf_features(
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, output_dir: Path, suffix: str = ""
):
    """
    TF-IDF特徴量を保存
    """
    print(f"\nTF-IDF特徴量保存中...（suffix={suffix}）")

    # Sparse matrixをdenseに変換（LightGBM用）
    X_train_dense = X_train.toarray().astype(np.float32)
    X_val_dense = X_val.toarray().astype(np.float32)
    X_test_dense = X_test.toarray().astype(np.float32)

    # 保存
    np.save(output_dir / f"X_train_tfidf{suffix}.npy", X_train_dense)
    np.save(output_dir / f"y_train_tfidf{suffix}.npy", y_train)
    np.save(output_dir / f"X_val_tfidf{suffix}.npy", X_val_dense)
    np.save(output_dir / f"y_val_tfidf{suffix}.npy", y_val)
    np.save(output_dir / f"X_test_tfidf{suffix}.npy", X_test_dense)
    np.save(output_dir / f"y_test_tfidf{suffix}.npy", y_test)

    # 語彙を保存
    np.save(output_dir / f"tfidf_vocabulary{suffix}.npy", vectorizer.get_feature_names_out())

    # Vectorizerを保存
    joblib.dump(vectorizer, output_dir / f"tfidf_vectorizer{suffix}.joblib")

    print(f"保存完了: {output_dir}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("YouTubeチャットモデレータモデル - TF-IDF特徴量抽出")
    print("=" * 60)

    INPUT_DIR = Path("outputs/features")
    OUTPUT_DIR = Path("outputs/features")

    # 1. データ読み込み
    print("\nデータ読み込み中...")
    train_df = pd.read_parquet(INPUT_DIR / "train_raw.parquet")
    val_df = pd.read_parquet(INPUT_DIR / "val_raw.parquet")
    test_df = pd.read_parquet(INPUT_DIR / "test_raw.parquet")

    train_texts = train_df["body"].fillna("").tolist()
    val_texts = val_df["body"].fillna("").tolist()
    test_texts = test_df["body"].fillna("").tolist()

    y_train = train_df["is_anomaly"].values.astype(np.int64)
    y_val = val_df["is_anomaly"].values.astype(np.int64)
    y_test = test_df["is_anomaly"].values.astype(np.int64)

    # 2. TF-IDF特徴量生成（複数設定で比較）

    # 設定A: 軽量（500次元、ユニグラム+バイグラム）
    X_train_a, X_val_a, X_test_a, vectorizer_a = create_tfidf_features(
        train_texts, val_texts, test_texts, max_features=500, ngram_range=(1, 2)
    )
    save_tfidf_features(
        X_train_a,
        X_val_a,
        X_test_a,
        y_train,
        y_val,
        y_test,
        vectorizer_a,
        OUTPUT_DIR,
        suffix="_500",
    )

    # 設定B: 中間（1000次元、ユニグラム+バイグラム）
    X_train_b, X_val_b, X_test_b, vectorizer_b = create_tfidf_features(
        train_texts, val_texts, test_texts, max_features=1000, ngram_range=(1, 2)
    )
    save_tfidf_features(
        X_train_b,
        X_val_b,
        X_test_b,
        y_train,
        y_val,
        y_test,
        vectorizer_b,
        OUTPUT_DIR,
        suffix="_1000",
    )

    # 設定C: 重い（2000次元、ユニグラム+バイグラム+トリグラム）
    X_train_c, X_val_c, X_test_c, vectorizer_c = create_tfidf_features(
        train_texts, val_texts, test_texts, max_features=2000, ngram_range=(1, 3)
    )
    save_tfidf_features(
        X_train_c,
        X_val_c,
        X_test_c,
        y_train,
        y_val,
        y_test,
        vectorizer_c,
        OUTPUT_DIR,
        suffix="_2000",
    )

    print("\n" + "=" * 60)
    print("TF-IDF特徴量抽出完了")
    print("=" * 60)
    print("\n生成されたファイル:")
    print("  - tfidf_500: 軽量版（500次元）")
    print("  - tfidf_1000: 標準版（1000次元）")
    print("  - tfidf_2000: 高精度版（2000次元）")


if __name__ == "__main__":
    main()
