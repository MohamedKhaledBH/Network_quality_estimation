from pathlib import Path
import json
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.signal import stft

MODEL_PATH = Path("artifacts/deepsig_model.keras")
METADATA_PATH = Path("artifacts/deepsig_model_metadata.json")
DEFAULT_CLASS_NAMES = ["Poor", "Average", "Good"]

app = FastAPI(title="DeepSig Network Quality API", version="1.0.0")
model: Optional[tf.keras.Model] = None
class_names = DEFAULT_CLASS_NAMES


class PredictRequest(BaseModel):
    iq_signal: list = Field(
        ..., description="IQ sample as a 1D list, a 2xN matrix, or an Nx2 matrix"
    )


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: dict[str, float]


def fast_spectrogram(iq_signal: np.ndarray) -> np.ndarray:
    iq_signal = np.asarray(iq_signal)

    if iq_signal.ndim == 1:
        complex_signal = iq_signal.astype(np.complex64)
    elif iq_signal.ndim == 2:
        if iq_signal.shape[0] == 2 and iq_signal.shape[1] != 2:
            complex_signal = iq_signal[0] + 1j * iq_signal[1]
        elif iq_signal.shape[-1] == 2:
            complex_signal = iq_signal[:, 0] + 1j * iq_signal[:, 1]
        else:
            raise ValueError(f"Unexpected IQ sample shape: {iq_signal.shape}")
    else:
        raise ValueError(f"Unexpected IQ sample rank: {iq_signal.ndim}")

    complex_signal = np.asarray(complex_signal).reshape(-1)
    nperseg = min(64, complex_signal.size)
    noverlap = min(48, max(0, nperseg - 1))

    _, _, zxx = stft(complex_signal, nperseg=nperseg, noverlap=noverlap)
    spectrum = np.abs(zxx)
    spectrum = np.log1p(spectrum)

    s_min = spectrum.min()
    s_max = spectrum.max()
    if s_max > s_min:
        spectrum = (spectrum - s_min) / (s_max - s_min)

    return cv2.resize(spectrum.astype("float32"), (128, 128))


def preprocess_input(iq_signal: list) -> np.ndarray:
    spectrogram = fast_spectrogram(np.asarray(iq_signal))
    spectrogram = np.repeat(spectrogram[..., np.newaxis], 3, axis=-1)
    return np.expand_dims(spectrogram, axis=0)


@app.on_event("startup")
def load_model() -> None:
    global model, class_names

    if not MODEL_PATH.exists():
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())
        class_names = metadata.get("class_names", DEFAULT_CLASS_NAMES)


@app.get("/health")
def health() -> dict[str, str]:
    if model is not None:
        return {"status": "ok", "model": MODEL_PATH.name}
    if MODEL_PATH.exists():
        return {"status": "ok", "model": "loaded_at_runtime_failed"}
    return {"status": "ok", "model": "missing_artifact"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        batch = preprocess_input(request.iq_signal)
        probabilities = model.predict(batch, verbose=0)[0]
        class_id = int(np.argmax(probabilities))
        probability_map = {
            class_names[index]: float(probabilities[index])
            for index in range(len(class_names))
        }
        return PredictResponse(
            class_id=class_id,
            class_name=class_names[class_id],
            probabilities=probability_map,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
