"""
YouTubeチャットモデレータモデル用データ前処理

処理内容:
1. データ読み込み（層化サンプリング済み）
2. ノイズ除去（flagged + nonflagged の両方に含まれるユーザーを除外）
3. 独立メッセージとしてデータを準備
4. ユーザーIDによるtrain/val/test分割
5. 分割済みデータの保存

注意:
- 各メッセージを独立に扱う（時系列要素なし）
- ユーザーIDは分割に使用するが、特徴量には含めない
"""

from pathlib import Path

import numpy as np
import pandas as pd

# パス設定
DATA_DIR = Path("data/raw/sensai-complete")
OUTPUT_DIR = Path("outputs/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_stratified_sample() -> pd.DataFrame:
    """
    層化サンプリング済みデータを読み込む

    Returns:
        サンプリング済みデータフレーム
    """
    # 前回作成したEDAノートブックと同様にサンプリング
    # ここでは簡易的に全体データの10%をサンプリング
    # 実際の実装では前回のstratified_sampling関数を使用

    print("データ読み込み中...")
    flagged_files = sorted(DATA_DIR.glob("chats_flagged_*.parquet"))
    nonflag_files = sorted(DATA_DIR.glob("chats_nonflag_*.parquet"))

    sampled_dfs = []
    samples_per_month = 500  # 各月500件ずつ（合計約19,500件）

    for f_flag, f_non in zip(flagged_files, nonflag_files):
        month = f_flag.stem.split("_")[-1]

        # flaggedデータ
        flagged_df = pd.read_parquet(f_flag)
        flagged_hidden = flagged_df[flagged_df["label"] == "hidden"]
        flagged_deleted = flagged_df[flagged_df["label"] == "deleted"]

        # nonflaggedデータ
        nonflag_df = pd.read_parquet(f_non)

        # サンプリング
        # 月あたりのサンプル件数(samples_per_month)を
        # - hidden + deleted（異常クラス）
        # - nonflagged（正常クラス）
        # に分割してサンプリングする
        anomaly_budget = samples_per_month // 2  # hidden + deleted 用
        non_anomaly_budget = samples_per_month - anomaly_budget  # nonflagged 用

        # hidden / deleted で異常クラスの予算をさらに分割
        n_hidden = anomaly_budget // 2
        n_deleted = anomaly_budget - n_hidden  # 端数が出た場合はこちらに寄せる

        # hidden
        if len(flagged_hidden) > n_hidden:
            sampled_hidden = flagged_hidden.sample(n=n_hidden, random_state=42).copy()
        else:
            sampled_hidden = flagged_hidden.copy()
        sampled_hidden["month"] = month
        sampled_hidden["is_anomaly"] = 1
        sampled_dfs.append(sampled_hidden)

        # deleted
        if len(flagged_deleted) > n_deleted:
            sampled_deleted = flagged_deleted.sample(n=n_deleted, random_state=42).copy()
        else:
            sampled_deleted = flagged_deleted.copy()
        sampled_deleted["month"] = month
        sampled_deleted["is_anomaly"] = 1
        sampled_dfs.append(sampled_deleted)

        # nonflagged（正常クラス）
        if len(nonflag_df) > non_anomaly_budget:
            sampled_nonflag = nonflag_df.sample(n=non_anomaly_budget, random_state=42).copy()
        else:
            sampled_nonflag = nonflag_df.copy()
        sampled_nonflag["month"] = month
        sampled_nonflag["is_anomaly"] = 0
        sampled_dfs.append(sampled_nonflag)

    df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"サンプリング完了: {len(df):,}件")
    return df


def remove_noise_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    ノイズとなるユーザーを除外

    除外条件:
    - flagged（hidden/deleted）かつ nonflagged の両方に含まれるユーザー

    Args:
        df: 入力データフレーム

    Returns:
        ノイズ除去後のデータフレーム
    """
    print("\nノイズ除去中...")

    # 各ユーザーのラベルを確認
    user_labels = df.groupby("authorName")["is_anomaly"].agg(["min", "max"])

    # flagged（is_anomaly=1）と nonflagged（is_anomaly=0）の両方を持つユーザー
    ambiguous_users = user_labels[(user_labels["min"] == 0) & (user_labels["max"] == 1)].index

    print(
        f"曖昧なユーザー数: {len(ambiguous_users):,} ({len(ambiguous_users) / df['authorName'].nunique() * 100:.2f}%)"
    )

    # 除外
    df_clean = df[~df["authorName"].isin(ambiguous_users)].copy()

    print(f"ノイズ除去後: {len(df_clean):,}件（{len(df_clean) / len(df) * 100:.1f}%）")
    print(f"除去されたレコード: {len(df) - len(df_clean):,}件")

    return df_clean


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ユーザーIDでグループ化し、層化分割を行う

    同じユーザーが複数のセットに含まれるのを防ぎ、
    かつ各セットで異常クラスの比率を維持する

    Args:
        df: 入力データフレーム
        train_ratio: 訓練データ比率
        val_ratio: 検証データ比率
        test_ratio: テストデータ比率
        random_state: 乱数シード

    Returns:
        (train_df, val_df, test_df): 分割済みデータフレーム
    """
    print("\nデータ分割中（層化 + ユーザーグループ）...")
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "比率の合計が1.0である必要があります"
    )

    # ユーザーごとのラベル（層化用: 1つでも異常があれば異常ユーザーとみなす）
    user_labels = df.groupby("authorName")["is_anomaly"].max().reset_index()

    # 異常ユーザーと正常ユーザーを分離
    anomaly_users = user_labels[user_labels["is_anomaly"] == 1]["authorName"].values
    normal_users = user_labels[user_labels["is_anomaly"] == 0]["authorName"].values

    # 各クラスでユーザーをシャッフル
    np.random.seed(random_state)
    np.random.shuffle(anomaly_users)
    np.random.shuffle(normal_users)

    # 各クラスをtrain/val/testに分割
    n_anomaly = len(anomaly_users)
    n_normal = len(normal_users)

    # 異常ユーザーの分割
    n_anomaly_test = int(n_anomaly * test_ratio)
    n_anomaly_val = int(n_anomaly * val_ratio)

    anomaly_test_users = anomaly_users[:n_anomaly_test]
    anomaly_val_users = anomaly_users[n_anomaly_test:n_anomaly_test + n_anomaly_val]
    anomaly_train_users = anomaly_users[n_anomaly_test + n_anomaly_val:]

    # 正常ユーザーの分割
    n_normal_test = int(n_normal * test_ratio)
    n_normal_val = int(n_normal * val_ratio)

    normal_test_users = normal_users[:n_normal_test]
    normal_val_users = normal_users[n_normal_test:n_normal_test + n_normal_val]
    normal_train_users = normal_users[n_normal_test + n_normal_val:]

    # 結合
    train_users = np.concatenate([anomaly_train_users, normal_train_users])
    val_users = np.concatenate([anomaly_val_users, normal_val_users])
    test_users = np.concatenate([anomaly_test_users, normal_test_users])

    # ユーザーIDで元のデータを分割
    train_df = df[df["authorName"].isin(train_users)].copy()
    val_df = df[df["authorName"].isin(val_users)].copy()
    test_df = df[df["authorName"].isin(test_users)].copy()

    print(
        f"Train: {len(train_df):,}件 ({len(train_df) / len(df) * 100:.1f}%) - {len(train_users):,}ユーザー"
    )
    print(
        f"Val:   {len(val_df):,}件 ({len(val_df) / len(df) * 100:.1f}%) - {len(val_users):,}ユーザー"
    )
    print(
        f"Test:  {len(test_df):,}件 ({len(test_df) / len(df) * 100:.1f}%) - {len(test_users):,}ユーザー"
    )

    # クラス分布の確認
    print("\nクラス分布:")
    for name, data in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        anomaly_ratio = data["is_anomaly"].mean()
        print(
            f"  {name}: 異常率 = {anomaly_ratio:.3f} ({data['is_anomaly'].sum():,}/{len(data):,})"
        )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path
) -> None:
    """
    分割済みデータを保存

    Args:
        train_df: 訓練データ
        val_df: 検証データ
        test_df: テストデータ
        output_dir: 出力ディレクトリ
    """
    print("\nデータ保存中...")

    # 保存（Parquet形式）
    train_df.to_parquet(output_dir / "train_raw.parquet", index=False)
    val_df.to_parquet(output_dir / "val_raw.parquet", index=False)
    test_df.to_parquet(output_dir / "test_raw.parquet", index=False)

    # メタデータ保存
    metadata = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_users": train_df["authorName"].nunique(),
        "val_users": val_df["authorName"].nunique(),
        "test_users": test_df["authorName"].nunique(),
    }

    import json

    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"保存完了: {output_dir}")
    print(f"  train_raw.parquet: {len(train_df):,}件")
    print(f"  val_raw.parquet: {len(val_df):,}件")
    print(f"  test_raw.parquet: {len(test_df):,}件")


def main():
    """メイン処理"""
    print("=" * 60)
    print("YouTubeチャットモデレータモデル - データ前処理")
    print("=" * 60)

    # 1. データ読み込み
    df = load_stratified_sample()

    # 2. ノイズ除去
    df_clean = remove_noise_users(df)

    # 3. データ分割
    train_df, val_df, test_df = split_data(df_clean)

    # 4. 保存
    save_splits(train_df, val_df, test_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("前処理完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
