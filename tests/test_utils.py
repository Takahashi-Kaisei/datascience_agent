import numpy as np

from datascience_agent.utils import calculate_mean


def test_calculate_mean_normal() -> None:
    data = np.array([1, 2, 3, 4, 5])
    result = calculate_mean(data)
    assert result == 3.0


def test_calculate_mean_empty() -> None:
    data = np.array([])
    result = calculate_mean(data)
    assert result is None


def test_calculate_mean_single() -> None:
    data = np.array([42.0])
    result = calculate_mean(data)
    assert result == 42.0


def test_calculate_mean_negative() -> None:
    data = np.array([-1, -2, -3])
    result = calculate_mean(data)
    assert result == -2.0
