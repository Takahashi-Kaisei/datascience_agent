"""
YouTubeチャットモデレータモデル - モデル学習と評価

処理内容:
1. LightGBMモデルの学習
2. is_unbalance vs ダウンサンプリングの比較
3. 閾値最適化（Precision >= 0.90を目標）
4. 5-Fold Cross Validation
5. 最終モデルの保存

Complexity: O(n * f * t) where n = samples, f = features, t = trees
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


class ModeratorModel:
    """
    YouTubeチャットモデレータモデル

    異常検知（flaggedメッセージの検出）を行う
    偽陽性（False Positive）を減らすことを重視
    """

    def __init__(self, params: dict = None):
        """
        モデル初期化

        Args:
            params: LightGBMパラメータ（デフォルト値使用可能）
        """
        self.params = params or self._get_default_params()
        self.model = None
        self.threshold = 0.5  # デフォルト閾値

    def _get_default_params(self) -> dict:
        """デフォルトパラメータ"""
        return {
            "objective": "binary",
            "metric": "None",  # カスタム評価で制御
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_precision: float = 0.90,
    ) -> dict:
        """
        モデルを学習し、最適な閾値を探索

        Args:
            X_train: 訓練データ特徴量
            y_train: 訓練データラベル
            X_val: 検証データ特徴量
            y_val: 検証データラベル
            target_precision: 目標適合率

        Returns:
            学習結果の辞書
        """
        # LightGBM Dataset作成
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 学習
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)],
        )

        # 最適な閾値を探索
        self.threshold = self._optimize_threshold(
            y_val, self.model.predict(X_val), target_precision
        )

        # 評価
        metrics = self.evaluate(X_val, y_val)

        return {
            "threshold": self.threshold,
            "best_iteration": self.model.best_iteration,
            "metrics": metrics,
        }

    def _optimize_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray, target_precision: float
    ) -> float:
        """
        目標適合率を達成する最適な閾値を探索

        Args:
            y_true: 真のラベル
            y_proba: 予測確率
            target_precision: 目標適合率

        Returns:
            最適な閾値
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # target_precision以上のインデックスを特定
        # precisions/recalls は thresholds より1要素長いため、thresholds の範囲内に制限
        valid_indices = np.where(precisions >= target_precision)[0]
        valid_indices = valid_indices[valid_indices < len(thresholds)]

        if len(valid_indices) == 0:
            # 目標達成不可の場合は0.5を返す
            print(f"警告: 目標適合率{target_precision}が達成できません。デフォルト閾値0.5を使用。")
            return 0.5

        # F0.5-score（Precision重視）が最大となる閾値を選択
        f05_scores = []
        for idx in valid_indices:
            p = precisions[idx]
            r = recalls[idx]
            if p + r > 0:
                f05 = (1 + 0.5**2) * (p * r) / (0.5**2 * p + r)
                f05_scores.append((f05, idx))

        if not f05_scores:
            return 0.5

        best_idx = max(f05_scores, key=lambda x: x[0])[1]
        optimal_threshold = thresholds[best_idx]

        return optimal_threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（閾値適用済み）"""
        if self.model is None:
            raise ValueError("モデルが学習されていません。fit()を先に実行してください。")

        y_proba = self.model.predict(X)
        return (y_proba >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """予測確率"""
        if self.model is None:
            raise ValueError("モデルが学習されていません。fit()を先に実行してください。")

        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        モデルを評価

        Args:
            X: 特徴量
            y_true: 真のラベル

        Returns:
            評価指標の辞書
        """
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
            "f05": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "average_precision": average_precision_score(y_true, y_proba),
        }

        return metrics

    def save(self, output_dir: Path, model_name: str):
        """モデルを保存"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # モデル保存
        joblib.dump(self.model, output_dir / f"{model_name}.joblib")

        # パラメータと閾値を保存
        metadata = {
            "params": self.params,
            "threshold": self.threshold,
        }
        with open(output_dir / f"{model_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"モデル保存完了: {output_dir / model_name}")


def cross_validate(
    X: np.ndarray, y: np.ndarray, params: dict, n_splits: int = 5, target_precision: float = 0.90
) -> dict:
    """
    5-Fold Cross Validationを実行

    Args:
        X: 特徴量（訓練+検証データ）
        y: ラベル
        params: LightGBMパラメータ
        n_splits: 分割数
        target_precision: 目標適合率

    Returns:
        CV結果の辞書
    """
    print(f"\n{n_splits}-Fold Cross Validation実行中...")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\nFold {fold_idx + 1}/{n_splits}")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = ModeratorModel(params)
        result = model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, target_precision)

        metrics = result["metrics"]
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F0.5: {metrics['f05']:.4f}")
        print(f"  Threshold: {result['threshold']:.4f}")

        fold_results.append(
            {"fold": fold_idx + 1, "threshold": result["threshold"], "metrics": metrics}
        )

    # 平均スコアを計算
    avg_metrics = {
        "precision": np.mean([r["metrics"]["precision"] for r in fold_results]),
        "recall": np.mean([r["metrics"]["recall"] for r in fold_results]),
        "f05": np.mean([r["metrics"]["f05"] for r in fold_results]),
        "roc_auc": np.mean([r["metrics"]["roc_auc"] for r in fold_results]),
    }

    print("\nCV平均スコア:")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f}")
    print(f"  F0.5: {avg_metrics['f05']:.4f}")

    return {"fold_results": fold_results, "avg_metrics": avg_metrics}


def compare_approaches(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_name: str,
) -> dict:
    """
    is_unbalance vs ダウンサンプリングを比較

    Args:
        X_train, y_train: 訓練データ
        X_val, y_val: 検証データ
        feature_name: 特徴量セット名（表示用）

    Returns:
        比較結果の辞書
    """
    print(f"\n{'=' * 60}")
    print(f"{feature_name} - アプローチ比較")
    print(f"{'=' * 60}")

    results = {}

    # 1. is_unbalance=True（自動調整）
    print("\n[アプローチA] is_unbalance=True（自動調整）")
    params_auto = {
        "objective": "binary",
        "metric": "None",
        "is_unbalance": True,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
    }

    model_auto = ModeratorModel(params_auto)
    result_auto = model_auto.fit(X_train, y_train, X_val, y_val, target_precision=0.90)
    metrics_auto = result_auto["metrics"]

    print(f"  Precision: {metrics_auto['precision']:.4f}")
    print(f"  Recall: {metrics_auto['recall']:.4f}")
    print(f"  F0.5: {metrics_auto['f05']:.4f}")
    print(f"  Threshold: {result_auto['threshold']:.4f}")

    results["is_unbalance"] = {
        "model": model_auto,
        "metrics": metrics_auto,
        "threshold": result_auto["threshold"],
    }

    # 2. ダウンサンプリング（1:1等比率）
    print("\n[アプローチB] ダウンサンプリング（1:1等比率）")

    # ダウンサンプリング
    n_anomaly = np.sum(y_train == 1)
    n_normal = np.sum(y_train == 0)
    anomaly_indices = np.where(y_train == 1)[0]
    normal_indices = np.where(y_train == 0)[0]

    # 少数クラスに合わせてサンプリング
    n_min = min(n_anomaly, n_normal)

    # 異常クラス・正常クラスを少数クラス数に揃えてサンプリング
    np.random.seed(42)
    if n_anomaly > n_min:
        sampled_anomaly_indices = np.random.choice(anomaly_indices, size=n_min, replace=False)
    else:
        sampled_anomaly_indices = anomaly_indices

    if n_normal > n_min:
        sampled_normal_indices = np.random.choice(normal_indices, size=n_min, replace=False)
    else:
        sampled_normal_indices = normal_indices

    # 結合
    downsampled_indices = np.concatenate([sampled_anomaly_indices, sampled_normal_indices])
    np.random.shuffle(downsampled_indices)

    X_train_down = X_train[downsampled_indices]
    y_train_down = y_train[downsampled_indices]

    print(
        f"  ダウンサンプリング後: {len(X_train_down)}件（異常: {np.sum(y_train_down == 1)}, 正常: {np.sum(y_train_down == 0)}）"
    )

    params_down = {
        "objective": "binary",
        "metric": "None",
        "is_unbalance": False,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
    }

    model_down = ModeratorModel(params_down)
    result_down = model_down.fit(X_train_down, y_train_down, X_val, y_val, target_precision=0.90)
    metrics_down = result_down["metrics"]

    print(f"  Precision: {metrics_down['precision']:.4f}")
    print(f"  Recall: {metrics_down['recall']:.4f}")
    print(f"  F0.5: {metrics_down['f05']:.4f}")
    print(f"  Threshold: {result_down['threshold']:.4f}")

    results["downsampling"] = {
        "model": model_down,
        "metrics": metrics_down,
        "threshold": result_down["threshold"],
    }

    # 最適アプローチの選択（F0.5-scoreベース）
    best_approach = "is_unbalance" if metrics_auto["f05"] > metrics_down["f05"] else "downsampling"
    print(
        f"\n最適アプローチ: {best_approach}（F0.5: {results[best_approach]['metrics']['f05']:.4f}）"
    )

    return results


def main():
    """メイン処理"""
    print("=" * 60)
    print("YouTubeチャットモデレータモデル - モデル学習")
    print("=" * 60)

    FEATURE_DIR = Path("outputs/features")
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 手動特徴量でモデル学習
    print("\n[手動特徴量（18次元）]")
    X_train = np.load(FEATURE_DIR / "X_train.npy")
    y_train = np.load(FEATURE_DIR / "y_train.npy")
    X_val = np.load(FEATURE_DIR / "X_val.npy")
    y_val = np.load(FEATURE_DIR / "y_val.npy")

    results_manual = compare_approaches(X_train, y_train, X_val, y_val, "手動特徴量")

    # 最適モデルを保存
    best_manual = (
        results_manual["is_unbalance"]
        if results_manual["is_unbalance"]["metrics"]["f05"]
        > results_manual["downsampling"]["metrics"]["f05"]
        else results_manual["downsampling"]
    )
    best_manual["model"].save(MODEL_DIR, "moderator_model_manual")

    # 2. TF-IDF特徴量（500次元）でモデル学習
    print("\n[TF-IDF特徴量（500次元）]")
    X_train_tfidf = sparse.load_npz(FEATURE_DIR / "X_train_tfidf_500.npz")
    y_train_tfidf = np.load(FEATURE_DIR / "y_train_tfidf_500.npy")
    X_val_tfidf = sparse.load_npz(FEATURE_DIR / "X_val_tfidf_500.npz")
    y_val_tfidf = np.load(FEATURE_DIR / "y_val_tfidf_500.npy")

    results_tfidf = compare_approaches(
        X_train_tfidf, y_train_tfidf, X_val_tfidf, y_val_tfidf, "TF-IDF 500次元"
    )

    # 最適モデルを保存
    best_tfidf = (
        results_tfidf["is_unbalance"]
        if results_tfidf["is_unbalance"]["metrics"]["f05"]
        > results_tfidf["downsampling"]["metrics"]["f05"]
        else results_tfidf["downsampling"]
    )
    best_tfidf["model"].save(MODEL_DIR, "moderator_model_tfidf_500")

    # 3. 結果の保存
    comparison_results = {
        "manual_features": {
            "is_unbalance": results_manual["is_unbalance"]["metrics"],
            "downsampling": results_manual["downsampling"]["metrics"],
        },
        "tfidf_features": {
            "is_unbalance": results_tfidf["is_unbalance"]["metrics"],
            "downsampling": results_tfidf["downsampling"]["metrics"],
        },
    }

    with open(MODEL_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n" + "=" * 60)
    print("モデル学習完了")
    print("=" * 60)
    print("\n保存されたモデル:")
    print("  - moderator_model_manual: 手動特徴量モデル")
    print("  - moderator_model_tfidf_500: TF-IDF特徴量モデル")


if __name__ == "__main__":
    main()
