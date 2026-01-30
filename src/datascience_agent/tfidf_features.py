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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


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

    # TfidfVectorizerの初期化（ネイティブ文字N-gram）
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
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

    print("TF-IDF特徴量生成完了:")
    print(f"  Train: {X_train_tfidf.shape}")
    print(f"  Val: {X_val_tfidf.shape}")
    print(f"  Test: {X_test_tfidf.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def save_tfidf_features(
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, output_dir: Path, suffix: str = ""
):
    """
    TF-IDF特徴量を保存（疎行列形式でメモリ効率化）
    """
    print(f"\nTF-IDF特徴量保存中...（suffix={suffix}）")

    # 疎行列のまま保存（LightGBMは疎行列をサポート）
    sparse.save_npz(output_dir / f"X_train_tfidf{suffix}.npz", X_train.astype(np.float32))
    sparse.save_npz(output_dir / f"X_val_tfidf{suffix}.npz", X_val.astype(np.float32))
    sparse.save_npz(output_dir / f"X_test_tfidf{suffix}.npz", X_test.astype(np.float32))

    # ラベルは通常の配列で保存
    np.save(output_dir / f"y_train_tfidf{suffix}.npy", y_train)
    np.save(output_dir / f"y_val_tfidf{suffix}.npy", y_val)
    np.save(output_dir / f"y_test_tfidf{suffix}.npy", y_test)

    # 語彙を保存
    np.save(output_dir / f"tfidf_vocabulary{suffix}.npy", vectorizer.get_feature_names_out())

    # Vectorizerを保存
    joblib.dump(vectorizer, output_dir / f"tfidf_vectorizer{suffix}.joblib")

    print(f"保存完了: {output_dir}")
    print(f"  疎行列形式で保存（メモリ効率: Train sparsity = {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.2%}）")


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
