import numpy as np


def calculate_mean(data: np.ndarray) -> float | None:
    """
    配列の平均値を計算。

    Args:
        data: 数値配列。

    Returns:
        平均値。配列が空の場合はNone。

    Complexity:
        O(n)
    """
    if len(data) == 0:
        return None
    return float(np.mean(data))


def load_data(filepath: str) -> np.ndarray:
    """
    CSVファイルからデータを読み込む。

    Args:
        filepath: CSVファイルのパス。

    Returns:
        データ配列。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
    """
    return np.loadtxt(filepath, delimiter=",")
