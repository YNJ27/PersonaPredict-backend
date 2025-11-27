"""FastAPI backend that exposes a personality prediction endpoint."""

import os
import sys
from typing import Any, Literal
from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Repo root (where `transformation.py` and the model file live).
# Since this `main.py` is now located at the repository root, point to
# the same directory rather than the parent used previously in `backend/`.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Import transformation from the repo root. No sys.path hack required.
from transformation import trf, threshold  # noqa: E402
from fastapi.staticfiles import StaticFiles

def _yes_no_to_int(value: str) -> int:
    """Convert Yes/No strings (case-insensitive) to their numeric encodings."""
    normalized = value.strip().lower()
    return 1 if normalized == "yes" else 0


class PersonalityInput(BaseModel):
    """Schema that validates both ranges and datatypes for each input feature."""

    hours_spent_alone: float = Field(
        ...,
        ge=0,
        le=11,
        description="Daily hours spent alone (0-11).",
    )
    stage_fear: Literal["Yes", "No"] = Field(
        ...,
        description="Presence of stage fear (Yes/No).",
    )
    social_event_frequency: float = Field(
        ...,
        ge=0,
        le=10,
        description="Frequency of social events attended (0-10).",
    )
    going_outside_frequency: float = Field(
        ...,
        ge=0,
        le=7,
        description="Frequency of going outside (0-7).",
    )
    drained_after_socializing: Literal["Yes", "No"] = Field(
        ...,
        description="Feeling drained after socializing (Yes/No).",
    )
    close_friends_count: float = Field(
        ...,
        ge=0,
        le=15,
        description="Number of close friends (0-15).",
    )
    social_media_post_frequency: float = Field(
        ...,
        ge=0,
        le=10,
        description="Social media post frequency (0-10).",
    )

    @field_validator("stage_fear", "drained_after_socializing", mode="before")
    @classmethod
    def normalize_yes_no(cls, value: str) -> str:
        """Force Yes/No fields into capitalized strings regardless of casing."""
        lowered = value.strip().lower()
        if lowered not in {"yes", "no"}:
            raise ValueError("Value must be either 'Yes' or 'No'.")
        return "Yes" if lowered == "yes" else "No"

    def as_ordered_numeric_array(self) -> np.ndarray:
        """Return a single-sample numpy array in the exact model feature order."""
        ordered_values = [
            self.hours_spent_alone,
            _yes_no_to_int(self.stage_fear),
            self.social_event_frequency,
            self.going_outside_frequency,
            _yes_no_to_int(self.drained_after_socializing),
            self.close_friends_count,
            self.social_media_post_frequency,
        ]
        return np.array([ordered_values], dtype=float)


def load_model() -> Any:
    """Load the serialized stacking classifier only once per process."""
    model_path = os.path.join(REPO_ROOT, "stacking_classifier_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model file 'stacking_classifier_model.joblib' not found in repo root."
        )
    return joblib.load(model_path)

app = FastAPI(
    title="Persona Predictor API",
    description="Simple FastAPI backend that serves introvert/extrovert predictions.",
    version="1.0.0",
)

# Allow the React dev server (running on Vite) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

# If the frontend has been built into `frontend/dist`, serve it as static files
# so the same FastAPI process can deliver the UI in production.
frontend_dist = os.path.join(REPO_ROOT, "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


@app.get("/health")
def health_check() -> dict:
    """Simple readiness probe."""
    return {"status": "ok"}


@app.post("/predict")
def predict_personality(payload: PersonalityInput) -> dict:
    """Accept user inputs, run the ML pipeline, and return the personality label."""
    try:
        numeric_inputs = payload.as_ordered_numeric_array()
        # Try to transform using the provided transformer. Some transformers may
        # require fit() before transform(), so fall back to fit_transform when
        # necessary (this mirrors earlier behavior while being more robust).
        try:
            transformed = trf.transform(numeric_inputs)
        except Exception:
            transformed = trf.fit_transform(numeric_inputs)
        prediction_probs = model.predict_proba(transformed)
        predicted_flag = int((prediction_probs[:, 1] >= threshold).astype(int)[0])
    except Exception as exc:  # noqa: BLE001 - surface inference issues
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    label = "Introvert" if predicted_flag == 1 else "Extrovert"
    return {
        "prediction": predicted_flag,
        "label": label,
        "confidence": float(prediction_probs[0][predicted_flag]),
    }


#uvicorn main:app --reload --port 8000 ..........(for local testing)

#render deployed link: https://personapredict-backend.onrender.com/