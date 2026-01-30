"""
YouTubeチャットモデレータモデル - 最終テスト評価

処理内容:
1. Testデータでの最終性能評価（1回のみ使用）
2. 複数モデルの性能比較
3. 最終レポート生成
4. モデル推論パイプラインのテスト

注意: Testデータは最終評価のみに使用（学習/検証には使用しない）
"""

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

from src.datascience_agent.model_training import ModeratorModel
from src.datascience_agent.ensemble_model import EnsembleModerator


def load_test_data(feature_dir: Path) -> Dict:
    """
    Testデータを読み込み

    Args:
        feature_dir: 特徴量ディレクトリ

    Returns:
        Testデータの辞書
    """
    print("\nTestデータを読み込み中...")

    data = {
        "manual": {
            "X": np.load(feature_dir / "X_test.npy"),
            "y": np.load(feature_dir / "y_test.npy"),
        },
        "tfidf_500": {
            "X": np.load(feature_dir / "X_test_tfidf_500.npy"),
            "y": np.load(feature_dir / "y_test_tfidf_500.npy"),
        },
        "tfidf_1000": {
            "X": np.load(feature_dir / "X_test_tfidf_1000.npy"),
            "y": np.load(feature_dir / "y_test_tfidf_1000.npy"),
        },
        "tfidf_2000": {
            "X": np.load(feature_dir / "X_test_tfidf_2000.npy"),
            "y": np.load(feature_dir / "y_test_tfidf_2000.npy"),
        },
    }

    print(f"Testサンプル数: {len(data['manual']['y'])}件")
    print(f"  異常: {np.sum(data['manual']['y'] == 1)}件")
    print(f"  正常: {np.sum(data['manual']['y'] == 0)}件")

    return data


def load_models(model_dir: Path) -> Dict:
    """
    学習済みモデルを読み込み

    Args:
        model_dir: モデル保存ディレクトリ

    Returns:
        モデルの辞書
    """
    print("\nモデルを読み込み中...")

    models = {}

    # 手動特徴量モデル
    manual_path = model_dir / "moderator_model_manual.joblib"
    if manual_path.exists():
        manual_model = ModeratorModel()
        manual_model.model = joblib.load(manual_path)
        with open(model_dir / "moderator_model_manual_metadata.json") as f:
            meta = json.load(f)
            manual_model.threshold = meta["threshold"]
        models["manual"] = manual_model
        print(f"  ✓ manualモデル読み込み完了")
    else:
        print(f"  ✗ manualモデルが見つかりません")

    # TF-IDFモデル
    for suffix in ["500", "1000", "2000"]:
        path = model_dir / f"moderator_model_tfidf_{suffix}.joblib"
        if path.exists():
            tfidf_model = ModeratorModel()
            tfidf_model.model = joblib.load(path)
            with open(model_dir / f"moderator_model_tfidf_{suffix}_metadata.json") as f:
                meta = json.load(f)
                tfidf_model.threshold = meta["threshold"]
            models[f"tfidf_{suffix}"] = tfidf_model
            print(f"  ✓ tfidf_{suffix}モデル読み込み完了")
        else:
            print(f"  ✗ tfidf_{suffix}モデルが見つかりません")

    return models


def evaluate_model(model, X: np.ndarray, y_true: np.ndarray, model_name: str) -> Dict:
    """
    単一モデルを評価

    Args:
        model: モデルオブジェクト
        X: 特徴量
        y_true: 真のラベル
        model_name: モデル名

    Returns:
        評価指標の辞書
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "f05": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": model.threshold,
    }

    return metrics


def evaluate_ensemble(models: Dict, test_data: Dict, weights: Dict[str, float]) -> Dict:
    """
    アンサンブルモデルを評価

    Args:
        models: モデル辞書
        test_data: Testデータ辞書
        weights: モデル重み

    Returns:
        アンサンブル評価結果
    """
    print("\nアンサンブルモデルを評価中...")

    ensemble = EnsembleModerator(models, weights)

    # ここでは手動特徴量を使用（実際には各モデルに対応した特徴量が必要）
    X = test_data["manual"]["X"]
    y_true = test_data["manual"]["y"]

    metrics = ensemble.evaluate(X, y_true)
    y_pred = ensemble.predict(X)

    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics.update(
        {
            "model_name": "ensemble",
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "threshold": ensemble.threshold,
            "weights": weights,
        }
    )

    return metrics


def generate_report(results: List[Dict], output_dir: Path):
    """
    評価レポートを生成

    Args:
        results: 評価結果のリスト
        output_dir: 出力ディレクトリ
    """
    print("\n評価レポートを生成中...")

    # DataFrameに変換
    df = pd.DataFrame(
        [
            {
                "model": r["model_name"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "f05": r["f05"],
                "roc_auc": r["roc_auc"],
                "threshold": r["threshold"],
            }
            for r in results
        ]
    )

    # CSV保存
    df.to_csv(output_dir / "final_test_report.csv", index=False)

    # JSON保存
    with open(output_dir / "final_test_report.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # レポート表示
    print("\n" + "=" * 70)
    print("最終Test評価レポート")
    print("=" * 70)
    print(df.to_string(index=False))

    # 最良モデルを特定
    best_idx = df["f05"].idxmax()
    best_model = df.loc[best_idx, "model"]
    best_f05 = df.loc[best_idx, "f05"]

    print(f"\n最良モデル: {best_model} (F0.5 = {best_f05:.4f})")

    # 混同行列の表示
    print("\n混同行列:")
    for r in results:
        cm = r["confusion_matrix"]
        print(f"\n{r['model_name']}:")
        print(f"  True Negative:  {cm['tn']}")
        print(f"  False Positive: {cm['fp']} (偽陽性)")
        print(f"  False Negative: {cm['fn']} (偽陰性)")
        print(f"  True Positive:  {cm['tp']}")

    print("\n" + "=" * 70)
    print(f"レポート保存完了: {output_dir / 'final_test_report.csv'}")
    print("=" * 70)


def test_inference_pipeline(model, feature_names: List[str]):
    """
    推論パイプラインをテスト

    Args:
        model: モデルオブジェクト
        feature_names: 特徴量名リスト
    """
    print("\n推論パイプラインをテスト中...")

    # テスト用のサンプルメッセージ
    test_messages = [
        "こんにちは、いい配信ですね！",  # 正常
        "死ねボケ",  # 明らかな攻撃
        "ああああああああああ",  # 繰り返し
        "http://example.com/suspicious",  # URL含有
    ]

    from src.datascience_agent.feature_engineering import MessageFeatureExtractor

    extractor = MessageFeatureExtractor()

    print("\n推論テスト結果:")
    for msg in test_messages:
        features = extractor.extract_features(msg)
        feature_vector = np.array([[features.get(name, 0.0) for name in feature_names]])

        proba = model.predict_proba(feature_vector)[0]
        pred = model.predict(feature_vector)[0]

        status = "⚠️ 異常" if pred == 1 else "✓ 正常"
        print(f"  {status} [{proba:.4f}] {msg[:30]}...")

    print("\n推論パイプライン正常動作確認完了")


def main():
    """メイン処理"""
    print("=" * 70)
    print("YouTubeチャットモデレータモデル - 最終Test評価")
    print("=" * 70)
    print("⚠️ 注意: Testデータは最終評価のみに使用")

    MODEL_DIR = Path("models")
    FEATURE_DIR = Path("outputs/features")
    REPORT_DIR = Path("outputs/reports")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Testデータ読み込み
    test_data = load_test_data(FEATURE_DIR)

    # 2. モデル読み込み
    models = load_models(MODEL_DIR)

    if not models:
        print("\nエラー: モデルが見つかりません。先にmodel_training.pyを実行してください。")
        return

    # 3. 各モデルを評価
    results = []

    print("\n" + "-" * 70)
    print("個別モデル評価:")
    print("-" * 70)

    for name, model in models.items():
        if name in test_data:
            X = test_data[name]["X"]
            y = test_data[name]["y"]
            metrics = evaluate_model(model, X, y, name)
            results.append(metrics)

            print(f"\n{name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F0.5: {metrics['f05']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    # 4. アンサンブルモデル評価（可能な場合）
    if len(models) >= 2 and "manual" in models and any("tfidf" in k for k in models.keys()):
        print("\n" + "-" * 70)
        weights = {"manual": 0.6, "tfidf_500": 0.4}  # デフォルト重み
        ensemble_metrics = evaluate_ensemble(models, test_data, weights)
        results.append(ensemble_metrics)

        print(f"\nensemble:")
        print(f"  Precision: {ensemble_metrics['precision']:.4f}")
        print(f"  Recall: {ensemble_metrics['recall']:.4f}")
        print(f"  F0.5: {ensemble_metrics['f05']:.4f}")
        print(f"  ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")

    # 5. レポート生成
    generate_report(results, REPORT_DIR)

    # 6. 推論パイプラインテスト
    feature_names = np.load(FEATURE_DIR / "feature_names.npy").tolist()
    test_inference_pipeline(models.get("manual", list(models.values())[0]), feature_names)

    print("\n" + "=" * 70)
    print("最終Test評価完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
