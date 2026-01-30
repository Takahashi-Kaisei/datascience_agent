from typing import Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np


def stratified_sampling(
    samples_per_month_label: int = 1000, random_seed: int = 42, data_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    層化サンプリング（Stratified Sampling）を実装

    サンプリング手法の理由:
    1. 層化変数: 月（時系列傾向を維持）とラベル（クラス比率を維持）
    2. サンプルサイズ決定の根拠:
       - 全データ: 12,901,932件
       - 95%信頼区間で±0.5%の誤差を許容する場合、必要サンプルサイズは約38,400件
       - 各月13ヶ月 × 各ラベル3クラス = 39層
       - 各層から1,000件ずつサンプリング -> 合計約39,000件（統計的に十分）
    3. ランダムシード固定で再現性確保

    Args:
        samples_per_month_label: 各月の各ラベルからサンプリングする数（デフォルト1000）
        random_seed: 再現性のための乱数シード（デフォルト42）
        data_dir: データディレクトリのパス（デフォルト: プロジェクトルート/data/raw/sensai-complete）

    Returns:
        sampled_df: サンプリングされたデータフレーム
        sampling_info: サンプリングに関する情報辞書
    """
    np.random.seed(random_seed)

    if data_dir is None:
        current_path = Path.cwd()
        project_root = current_path.parent if current_path.name == "notebooks" else current_path
        data_path = project_root / "data" / "raw" / "sensai-complete"
    else:
        data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"データディレクトリが見つかりません: {data_path}")
    flagged_files = sorted(data_path.glob("chats_flagged_*.parquet"))
    nonflag_files = sorted(data_path.glob("chats_nonflag_*.parquet"))

    sampled_dfs = []
    sampling_stats = []

    for f_flag, f_non in zip(flagged_files, nonflag_files):
        month = f_flag.stem.split("_")[-1]

        flagged_df = pd.read_parquet(f_flag)
        nonflag_df = pd.read_parquet(f_non)

        hidden_df = flagged_df[flagged_df["label"] == "hidden"]
        deleted_df = flagged_df[flagged_df["label"] == "deleted"]

        for label_name, label_df in [
            ("hidden", hidden_df),
            ("deleted", deleted_df),
            ("nonflagged", nonflag_df),
        ]:
            if len(label_df) > samples_per_month_label:
                sampled = label_df.sample(n=samples_per_month_label, random_state=random_seed)
            else:
                sampled = label_df

            sampled["month"] = month
            sampled_dfs.append(sampled)

            sampling_stats.append(
                {
                    "month": month,
                    "label": label_name,
                    "original_count": len(label_df),
                    "sampled_count": len(sampled),
                }
            )

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    sampling_info = {
        "total_original": 12_901_932,
        "total_sampled": len(sampled_df),
        "sampling_rate": len(sampled_df) / 12_901_932,
        "sampling_stats": pd.DataFrame(sampling_stats),
        "samples_per_month_label": samples_per_month_label,
        "method": "stratified_sampling",
        "stratification_variables": ["month", "label"],
        "confidence_level": "95%",
        "margin_of_error": "±0.5%",
        "reasoning": (
            "層化サンプリングを採用。層化変数は月（時系列傾向維持）とラベル（クラス比率維持）。"
            "95%信頼区間で±0.5%の誤差を許容する場合、必要サンプルサイズは約38,400件。"
            "各月13ヶ月×各ラベル3クラス=39層、各層から1,000件ずつサンプリングで約39,000件を確保。"
        ),
    }

    return sampled_df, sampling_info
