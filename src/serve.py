# =============================================================================
# VAI TRO CUA FILE: API suy luan (Buoc 2) - chay tren VM duoi dang service systemd.
#   - Luc khoi dong: tai models/model.joblib tu cloud storage ve VM, nap vao bo nho.
#   - GET  /healthz : job Release cua GitHub Actions goi de xac nhan deploy thanh cong.
#   - POST /score   : nhan 10 dac trung, tra ve nhan thu nhap du doan.
# File nay KHONG chay trong CI; no duoc scp len VM mot lan (xem tasks/buoc-2.md, muc 2.6).
# =============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Ten bucket lay tu bien moi truong (dat trong file systemd service tren VM).
ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tai file model.joblib tu cloud storage ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc (duoc dat trong systemd service).
    """
    # Bao dam thu muc ~/models ton tai (download_to_filename khong tu tao thu muc).
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # TODO 1: Tao storage.Client() - tu doc credentials tu GOOGLE_APPLICATION_CREDENTIALS
    client = storage.Client()

    # TODO 2: Lay bucket va blob tuong ung (blob = 1 "file" trong bucket)
    bucket = client.bucket(ARTIFACT_BUCKET)
    blob = bucket.blob(MODEL_KEY)

    # TODO 3: Tai file model xuong may
    blob.download_to_filename(MODEL_PATH)

    # TODO 4: In thong bao thanh cong (xem duoc bang `journalctl -u income-api`)
    print("Model da duoc tai xuong tu cloud storage.")


# Chay khi module duoc import tuc luc server khoi dong: tai model roi nap vao RAM.
# Moi lan `systemctl restart income-api` (job Release) -> tai lai model MOI NHAT.
download_model()
model = joblib.load(MODEL_PATH)

# Ten nhan tuong ung voi gia tri du doan cua model.
LABELS = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    # TODO 5: Tra ve dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    # TODO 6: Kiem tra so luong dac trung; sai thi tra loi 400 (loi tu phia client).
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")

    # TODO 7: Goi model.predict([req.features]) - predict nhan danh sach cac mau
    # nen boc 1 mau trong list; ket qua la mang, lay phan tu dau [0].
    pred = int(model.predict([req.features])[0])

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    return {"prediction": pred, "label": LABELS[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
