"""
YouTubeチャットモデレータモデル - SHAP解釈性分析

処理内容:
1. 手動特徴量モデルのSHAP値計算
2. グローバル特徴量重要度の可視化
3. 個別予測の解釈（ローカル解釈）
4. 誤判定例の分析

Complexity: O(n * f * t) where n = samples, f = features, t = trees
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt


class SHAPExplainer:
    """
    LightGBMモデルのSHAP解釈性分析

    モデルの判断理由を可視化・分析
    """

    def __init__(self, model: lgb.Booster, feature_names: List[str]):
        """
        SHAP Explainer初期化

        Args:
            model: LightGBMモデル
            feature_names: 特徴量名のリスト
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        SHAP値を計算

        Args:
            X: 特徴量（サンプル数 x 特徴量数）

        Returns:
            SHAP値（サンプル数 x 特徴量数）
        """
        print(f"SHAP値を計算中... {X.shape[0]}サンプル")
        shap_values = self.explainer.shap_values(X)

        # shap_valuesがリストの場合（二値分類）、positiveクラスを選択
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 異常クラスのSHAP値

        return shap_values

    def global_feature_importance(
        self, X: np.ndarray, output_dir: Path, top_n: int = 20
    ) -> pd.DataFrame:
        """
        グローバル特徴量重要度を計算・可視化

        Args:
            X: 特徴量
            output_dir: 出力ディレクトリ
            top_n: 表示する上位特徴量数

        Returns:
            特徴量重要度のDataFrame
        """
        print(f"\nグローバル特徴量重要度を計算中...")

        shap_values = self.compute_shap_values(X)

        # 平均絶対SHAP値で重要度を計算
        importance = np.abs(shap_values).mean(axis=0)

        # DataFrameに整形
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        # 保存
        importance_df.to_csv(output_dir / "global_feature_importance.csv", index=False)

        # 可視化
        self._plot_global_importance(shap_values, X, output_dir, top_n)

        print(f"トップ{top_n}重要度:")
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def _plot_global_importance(
        self, shap_values: np.ndarray, X: np.ndarray, output_dir: Path, top_n: int = 20
    ):
        """グローバル重要度をプロット"""
        # Summary plot（Beeswarm plot）
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X, feature_names=self.feature_names, max_display=top_n, show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=top_n,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_bar_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  保存: {output_dir / 'shap_summary_plot.png'}")
        print(f"  保存: {output_dir / 'shap_bar_plot.png'}")

    def explain_single_prediction(self, X: np.ndarray, idx: int, output_dir: Path) -> Dict:
        """
        個別予測の解釈（ローカル解釈）

        Args:
            X: 特徴量
            idx: 解釈するサンプルのインデックス
            output_dir: 出力ディレクトリ

        Returns:
            予測理由の辞書
        """
        print(f"\nサンプル{idx}の予測を解釈中...")

        shap_values = self.compute_shap_values(X[idx : idx + 1])

        # Waterfall plot
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=X[idx],
                feature_names=self.feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_waterfall_sample_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 重要な特徴量を特定
        feature_importance = list(zip(self.feature_names, shap_values[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        top_positive = [(f, v) for f, v in feature_importance if v > 0][:3]
        top_negative = [(f, v) for f, v in feature_importance if v < 0][:3]

        explanation = {
            "sample_idx": idx,
            "base_value": float(self.explainer.expected_value),
            "prediction": float(self.model.predict(X[idx : idx + 1])[0]),
            "top_positive_features": [
                {"feature": f, "shap_value": float(v)} for f, v in top_positive
            ],
            "top_negative_features": [
                {"feature": f, "shap_value": float(v)} for f, v in top_negative
            ],
        }

        print(f"  予測値: {explanation['prediction']:.4f}")
        print(f"  ベース値: {explanation['base_value']:.4f}")
        print(f"  主な要因（異常寄与）:")
        for feat in top_positive:
            print(f"    - {feat[0]}: +{feat[1]:.4f}")

        return explanation

    def analyze_misclassifications(
        self, X: np.ndarray, y_true: np.ndarray, output_dir: Path, n_samples: int = 10
    ) -> Dict:
        """
        誤判定例を分析

        False PositiveとFalse Negativeを分析

        Args:
            X: 特徴量
            y_true: 真のラベル
            output_dir: 出力ディレクトリ
            n_samples: 分析するサンプル数

        Returns:
            誤判定分析結果の辞書
        """
        print(f"\n誤判定例を分析中...")

        # 予測
        y_proba = self.model.predict(X)
        y_pred = (y_proba >= 0.5).astype(int)

        # False Positive（偽陽性）: 正常を異常と予測
        fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]

        # False Negative（偽陰性）: 異常を正常と予測
        fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]

        print(f"  False Positive: {len(fp_indices)}件")
        print(f"  False Negative: {len(fn_indices)}件")

        results = {
            "false_positives": [],
            "false_negatives": [],
        }

        # False Positiveを分析
        if len(fp_indices) > 0:
            print(f"\n  False Positiveサンプルを分析（上位{n_samples}件）:")
            # 予測確率が高い順にソート
            fp_proba = y_proba[fp_indices]
            top_fp = fp_indices[np.argsort(fp_proba)[-n_samples:]]

            for idx in top_fp:
                explanation = self.explain_single_prediction(X, idx, output_dir)
                explanation["type"] = "false_positive"
                explanation["prediction_probability"] = float(y_proba[idx])
                results["false_positives"].append(explanation)

        # False Negativeを分析
        if len(fn_indices) > 0:
            print(f"\n  False Negativeサンプルを分析（上位{n_samples}件）:")
            # 予測確率が低い順にソート
            fn_proba = y_proba[fn_indices]
            top_fn = fn_indices[np.argsort(fn_proba)[:n_samples]]

            for idx in top_fn:
                explanation = self.explain_single_prediction(X, idx, output_dir)
                explanation["type"] = "false_negative"
                explanation["prediction_probability"] = float(y_proba[idx])
                results["false_negatives"].append(explanation)

        # 保存
        with open(output_dir / "misclassification_analysis.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results


def main():
    """メイン処理"""
    print("=" * 60)
    print("YouTubeチャットモデレータモデル - SHAP解釈性分析")
    print("=" * 60)

    MODEL_DIR = Path("models")
    FEATURE_DIR = Path("outputs/features")
    REPORT_DIR = Path("outputs/reports")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. モデル読み込み
    print("\nモデルを読み込み中...")
    model_path = MODEL_DIR / "moderator_model_manual.joblib"
    if not model_path.exists():
        print(f"モデルが見つかりません: {model_path}")
        print("先にmodel_training.pyを実行してください。")
        return

    model = joblib.load(model_path)

    # 2. 特徴量名読み込み
    feature_names = np.load(FEATURE_DIR / "feature_names.npy").tolist()

    # 3. 検証データ読み込み
    print("\n検証データを読み込み中...")
    X_val = np.load(FEATURE_DIR / "X_val.npy")
    y_val = np.load(FEATURE_DIR / "y_val.npy")

    # サンプリング（計算時間短縮のため）
    n_samples = min(1000, len(X_val))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_val), size=n_samples, replace=False)
    X_sample = X_val[sample_indices]
    y_sample = y_val[sample_indices]

    print(f"分析対象: {n_samples}サンプル")

    # 4. SHAP Explainer作成
    explainer = SHAPExplainer(model, feature_names)

    # 5. グローバル特徴量重要度
    print("\n" + "-" * 60)
    importance_df = explainer.global_feature_importance(X_sample, REPORT_DIR, top_n=20)

    # 6. 個別予測の解釈（いくつかのサンプル）
    print("\n" + "-" * 60)
    print("個別予測の解釈:")

    # 正常と異常のサンプルをいくつか選ぶ
    normal_indices = np.where(y_sample == 0)[0][:3]
    anomaly_indices = np.where(y_sample == 1)[0][:3]

    for idx in np.concatenate([normal_indices, anomaly_indices]):
        explanation = explainer.explain_single_prediction(X_sample, idx, REPORT_DIR)

    # 7. 誤判定分析
    print("\n" + "-" * 60)
    misclass_results = explainer.analyze_misclassifications(
        X_sample, y_sample, REPORT_DIR, n_samples=5
    )

    print("\n" + "=" * 60)
    print("SHAP解釈性分析完了")
    print("=" * 60)
    print(f"\n生成されたレポート:")
    print(f"  - {REPORT_DIR / 'global_feature_importance.csv'}")
    print(f"  - {REPORT_DIR / 'shap_summary_plot.png'}")
    print(f"  - {REPORT_DIR / 'shap_bar_plot.png'}")
    print(f"  - {REPORT_DIR / 'misclassification_analysis.json'}")


if __name__ == "__main__":
    main()
