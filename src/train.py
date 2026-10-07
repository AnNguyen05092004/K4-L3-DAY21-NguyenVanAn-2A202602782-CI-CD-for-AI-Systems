# =============================================================================
# VAI TRO CUA FILE: script huan luyen mo hinh (Buoc 1) - la "trai tim" cua pipeline.
#   - Chay cuc bo o Buoc 1 de thi nghiem nhieu bo sieu tham so, ghi vao MLflow.
#   - Chay trong GitHub Actions (job Train) o Buoc 2/3 de sinh ra:
#       outputs/report.json  -> job quality-gate doc f1_score tu day
#       models/model.joblib  -> duoc upload len cloud storage roi VM tai ve phuc vu
#   - Duoc tests/test_train.py goi truc tiep (import ham train) de kiem thu.
# =============================================================================
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    # Moi file CSV co 10 cot dac trung + cot "target" (0 = thu nhap thap, 1 = cao).
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    # X = tat ca cot tru "target"; y = chi cot "target".
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Moi lan chay train() = mot "run" trong MLflow. Moi thu log ben trong khoi
    # `with` se gan vao run do (de so sanh cac run tren MLflow UI).
    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so (n_estimators, learning_rate, max_depth)
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen GradientBoostingClassifier
        # random_state=42 co dinh tinh ngau nhien -> chay lai cho ket qua giong nhau.
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap holdout va tinh chi so
        # f1_score(y_eval, preds) mac dinh tinh cho LOP DUONG (target = 1).
        # KHONG truyen average="weighted"/"macro" vi se bi lop da so keo len cao.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # TODO 6: Ghi nhan chi so vao MLflow + luu ca model thanh artifact cua run
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh (se hien trong log cua GitHub Actions)
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # TODO 8: Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2 (buoc "Read report").
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({"f1_score": f1, "accuracy": acc}, f)

        # TODO 9: Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2, VM se tai ve de phuc vu.
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1 de noi goi ham (test, __main__) doc ket qua
    return f1


if __name__ == "__main__":
    # Doc sieu tham so tu params.yaml roi huan luyen. Muon thu bo tham so khac:
    # sua params.yaml roi chay lai `python src/train.py`.
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
