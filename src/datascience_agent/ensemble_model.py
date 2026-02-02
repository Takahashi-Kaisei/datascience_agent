"""
YouTubeチャットモデレータモデル - アンサンブルモデル

処理内容:
1. 複数モデルの予測を組み合わせて性能向上
2. 重み付き平均アンサンブル
3. スタッキング（メタ学習）
4. 最適なアンサンブル戦略の探索

Complexity: O(m * n) where m = models, n = samples
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EnsembleModerator:
    """
    YouTubeチャットモデレータのアンサンブルモデル

    複数のモデルの予測を組み合わせて精度向上
    """

    def __init__(self, models: Dict[str, object], weights: Dict[str, float] = None):
        """
        アンサンブルモデル初期化

        Args:
            models: モデル名 -> モデルオブジェクトの辞書
            weights: モデル名 -> 重みの辞書（Noneの場合は等重み）
        """
        self.models = models
        self.weights = weights or {name: 1.0 / len(models) for name in models.keys()}
        self.threshold = 0.5

    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        予測確率を計算

        Args:
            X: 特徴量
            model_name: 特定のモデル名（Noneの場合はアンサンブル）

        Returns:
            予測確率
        """
        if model_name is not None:
            return self.models[model_name].predict_proba(X)

        # アンサンブル予測（重み付き平均）
        ensemble_proba = np.zeros(len(X))
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            ensemble_proba += self.weights[name] * proba

        return ensemble_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（閾値適用済み）"""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def optimize_threshold(
        self, X: np.ndarray, y_true: np.ndarray, target_precision: float = 0.90
    ) -> float:
        """
        最適な閾値を探索

        Args:
            X: 特徴量
            y_true: 真のラベル
            target_precision: 目標適合率

        Returns:
            最適な閾値
        """
        from sklearn.metrics import precision_recall_curve

        y_proba = self.predict_proba(X)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # target_precision以上のインデックスを特定
        valid_indices = np.where(precisions >= target_precision)[0]

        if len(valid_indices) == 0:
            print(f"警告: 目標適合率{target_precision}が達成できません。デフォルト閾値0.5を使用。")
            self.threshold = 0.5
            return 0.5

        # F0.5-score（Precision重視）が最大となる閾値を選択
        best_f05 = 0
        best_threshold = 0.5

        for idx in valid_indices:
            p = precisions[idx]
            r = recalls[idx]
            if p + r > 0:
                f05 = (1 + 0.5**2) * (p * r) / (0.5**2 * p + r)
                if f05 > best_f05:
                    best_f05 = f05
                    best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

        self.threshold = best_threshold
        return best_threshold

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
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
        }

        return metrics

    def save(self, output_dir: Path, model_name: str):
        """モデルを保存"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 各モデルを保存
        for name, model in self.models.items():
            model.save(output_dir, f"{model_name}_{name}")

        # アンサンブル設定を保存
        ensemble_config = {
            "weights": self.weights,
            "threshold": self.threshold,
            "model_names": list(self.models.keys()),
        }

        with open(output_dir / f"{model_name}_ensemble_config.json", "w") as f:
            json.dump(ensemble_config, f, indent=2)

        print(f"アンサンブルモデル保存完了: {output_dir / model_name}")


def load_individual_models(model_dir: Path) -> Dict[str, object]:
    """
    個別モデルを読み込む

    Args:
        model_dir: モデル保存ディレクトリ

    Returns:
        モデル名 -> モデルオブジェクトの辞書
    """
    from .model_training import ModeratorModel

    models = {}

    # 手動特徴量モデル
    if (model_dir / "moderator_model_manual.joblib").exists():
        manual_model = ModeratorModel()
        manual_model.model = joblib.load(model_dir / "moderator_model_manual.joblib")
        with open(model_dir / "moderator_model_manual_metadata.json") as f:
            meta = json.load(f)
            manual_model.threshold = meta["threshold"]
        models["manual"] = manual_model

    # TF-IDFモデル
    for suffix in ["500", "1000", "2000"]:
        model_path = model_dir / f"moderator_model_tfidf_{suffix}.joblib"
        if model_path.exists():
            tfidf_model = ModeratorModel()
            tfidf_model.model = joblib.load(model_path)
            with open(model_dir / f"moderator_model_tfidf_{suffix}_metadata.json") as f:
                meta = json.load(f)
                tfidf_model.threshold = meta["threshold"]
            models[f"tfidf_{suffix}"] = tfidf_model

    return models


def optimize_ensemble_weights(
    models: Dict[str, object],
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_sets: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    アンサンブルの重みを最適化

    Args:
        models: モデル辞書
        X_val: 検証データ特徴量
        y_val: 検証データラベル
        feature_sets: モデル名 -> 特徴量セットの辞書

    Returns:
        最適な重みの辞書
    """
    print("\nアンサンブル重みを最適化中...")

    # 各モデルの予測確率を取得
    model_probas = {}
    for name, model in models.items():
        X = feature_sets[name]
        model_probas[name] = model.predict_proba(X)

    # グリッドサーチで最適な重みを探索
    best_f05 = 0
    best_weights = {}

    # シンプルな重みパターンを試行
    weight_patterns = [
        {"manual": 0.7, "tfidf_500": 0.3},
        {"manual": 0.6, "tfidf_500": 0.4},
        {"manual": 0.5, "tfidf_500": 0.5},
        {"manual": 0.4, "tfidf_500": 0.6},
    ]

    for weights in weight_patterns:
        # 重み付き平均
        ensemble_proba = np.zeros(len(y_val))
        for name, weight in weights.items():
            if name in model_probas:
                ensemble_proba += weight * model_probas[name]

        # 閾値0.5で評価（後で最適化）
        y_pred = (ensemble_proba >= 0.5).astype(int)
        f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=0)

        print(f"  重み{weights}: F0.5 = {f05:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_weights = weights

    print(f"\n最適な重み: {best_weights}（F0.5 = {best_f05:.4f}）")
    return best_weights


def main():
    """メイン処理"""
    print("=" * 60)
    print("YouTubeチャットモデレータモデル - アンサンブルモデル構築")
    print("=" * 60)

    MODEL_DIR = Path("models")
    FEATURE_DIR = Path("outputs/features")

    # 1. 個別モデルを読み込み
    print("\n個別モデルを読み込み中...")
    models = load_individual_models(MODEL_DIR)

    if not models:
        print("モデルが見つかりません。先にmodel_training.pyを実行してください。")
        return

    print(f"読み込んだモデル: {list(models.keys())}")

    # 2. 検証データを読み込み
    print("\n検証データを読み込み中...")
    feature_sets = {
        "manual": np.load(FEATURE_DIR / "X_val.npy"),
        "tfidf_500": np.load(FEATURE_DIR / "X_val_tfidf_500.npy"),
        "tfidf_1000": np.load(FEATURE_DIR / "X_val_tfidf_1000.npy"),
        "tfidf_2000": np.load(FEATURE_DIR / "X_val_tfidf_2000.npy"),
    }
    y_val = np.load(FEATURE_DIR / "y_val.npy")

    # 3. アンサンブル重みを最適化
    weights = optimize_ensemble_weights(models, None, y_val, feature_sets)

    # 4. アンサンブルモデルを作成
    ensemble = EnsembleModerator(models, weights)

    # 5. 閾値最適化
    print("\n閾値を最適化中...")
    # ここでは手動特徴量を使用（実際には各モデルの重み付き平均）
    X_val_manual = feature_sets["manual"]
    threshold = ensemble.optimize_threshold(X_val_manual, y_val, target_precision=0.90)
    print(f"最適な閾値: {threshold:.4f}")

    # 6. 評価
    print("\nアンサンブルモデルの評価:")
    metrics = ensemble.evaluate(X_val_manual, y_val)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F0.5: {metrics['f05']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    # 7. 個別モデルとの比較
    print("\n個別モデルとの比較:")
    for name, model in models.items():
        X = feature_sets[name] if name in feature_sets else X_val_manual
        metrics_single = model.evaluate(X, y_val)
        print(
            f"  {name}: Precision={metrics_single['precision']:.4f}, "
            f"F0.5={metrics_single['f05']:.4f}"
        )

    # 8. 保存
    ensemble.save(MODEL_DIR, "ensemble_moderator")

    print("\n" + "=" * 60)
    print("アンサンブルモデル構築完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
