from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Farmer Machine Learning API",
    description="Farmer Income Prediction and Farmer Clustering API",
    version="2.0.0",
)

# ============================================================
# PATHS
# main.py is expected at: repo/app/backend/main.py
# ============================================================

BASE_DIR = Path(__file__).resolve().parent          # app/backend
APP_DIR = BASE_DIR.parent                          # app
PROJECT_DIR = APP_DIR.parent                       # repository root
MODEL_DIR = APP_DIR / "Pkl.files"

REGRESSION_MODEL_PATH = MODEL_DIR / "best_regression_model.pkl"
REGRESSION_SCALER_PATH = MODEL_DIR / "scaler.pkl"
REGRESSION_ENCODER_PATH = MODEL_DIR / "encoder.pkl"

# Create this bundle using the offline training script described below.
CLUSTER_BUNDLE_PATH = MODEL_DIR / "clustering_bundle.pkl"

# Optional small JSON file containing precomputed cluster counts/means.
# Do NOT load the 187–189 MB training CSV in this web service.
CLUSTER_SUMMARY_PATH = MODEL_DIR / "cluster_summary.json"


# ============================================================
# LOAD REGRESSION ARTIFACTS
# ============================================================

def load_joblib_if_exists(path: Path):
    if not path.exists():
        print(f"Artifact not found: {path}")
        return None
    return joblib.load(path)


try:
    regression_model = load_joblib_if_exists(REGRESSION_MODEL_PATH)
    regression_scaler = load_joblib_if_exists(REGRESSION_SCALER_PATH)
    regression_encoder = load_joblib_if_exists(REGRESSION_ENCODER_PATH)
    print("Regression artifacts loaded.")
except Exception as exc:
    print(f"Regression artifact loading failed: {exc}")
    regression_model = None
    regression_scaler = None
    regression_encoder = None


# ============================================================
# REGRESSION INPUT / FEATURES
# ============================================================

class PredictionInput(BaseModel):
    farming_system: str = Field(..., description="Farming system used by the farmer")
    head_of_household_age: float = Field(..., ge=0)
    land_owned_hectares: float = Field(..., ge=0)
    fertilizer_used_kg_per_hectare: float = Field(..., ge=0)
    goats_number: float = Field(..., ge=0)
    sheep_number: float = Field(..., ge=0)
    livestock_eggs_per_week: float = Field(..., ge=0)
    livestock_milk_litres_per_week: float = Field(..., ge=0)


REGRESSION_CAT_COLS = ["farming_system"]
REGRESSION_NUM_COLS = [
    "head_of_household_age",
    "land_owned_hectares",
    "fertilizer_used_kg_per_hectare",
    "goats_number",
    "sheep_number",
    "livestock_eggs_per_week",
    "livestock_milk_litres_per_week",
]
REGRESSION_NEW_NUM_COLS = REGRESSION_NUM_COLS + [
    "total_livestock_count",
    "fertilizer_intensity",
    "weekly_animal_yield",
    "is_senior_farmer",
]
REGRESSION_SKEWED_COLS = [
    "land_owned_hectares",
    "fertilizer_used_kg_per_hectare",
    "fertilizer_intensity",
]


def create_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["total_livestock_count"] = (
        result["goats_number"] + result["sheep_number"]
    )
    result["fertilizer_intensity"] = (
        result["fertilizer_used_kg_per_hectare"]
        / (result["land_owned_hectares"] + 0.01)
    )
    result["weekly_animal_yield"] = (
        result["livestock_eggs_per_week"]
        + result["livestock_milk_litres_per_week"]
    )
    result["is_senior_farmer"] = np.where(
        result["head_of_household_age"] >= 60, 1, 0
    )
    return result


@app.post("/predict")
def predict_income(data: PredictionInput):
    if any(x is None for x in (
        regression_model, regression_scaler, regression_encoder
    )):
        raise HTTPException(
            status_code=503,
            detail="Regression artifacts are missing or failed to load.",
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])
        features = create_regression_features(input_df)

        for col in REGRESSION_SKEWED_COLS:
            features[col] = np.log1p(np.maximum(0, features[col]))

        encoded = regression_encoder.transform(features[REGRESSION_CAT_COLS])
        encoded_names = regression_encoder.get_feature_names_out(
            REGRESSION_CAT_COLS
        )
        encoded_df = pd.DataFrame(
            encoded, columns=encoded_names, index=features.index
        )

        prepared = pd.concat(
            [features[REGRESSION_NEW_NUM_COLS], encoded_df], axis=1
        )

        # Preserve the exact training feature order if the scaler exposes it.
        expected = getattr(regression_scaler, "feature_names_in_", None)
        if expected is not None:
            prepared = prepared.reindex(columns=list(expected), fill_value=0)

        scaled = regression_scaler.transform(prepared)
        predicted_log = regression_model.predict(scaled)
        predicted_income = float(np.expm1(predicted_log[0]))

        return {
            "status": "success",
            "predicted_income_ngn": round(predicted_income, 2),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        )


# ============================================================
# CLUSTER ARTIFACTS
#
# clustering_bundle.pkl must be a dictionary with these keys:
# model, scaler, feature_cols, num_cols, binary_cat_cols,
# multi_cat_cols, binary_encoders, ordinal_encoder,
# numeric_medians, iqr_bounds, cluster_counts, statistics
#
# cluster_counts and statistics are optional. See process below.
# ============================================================

cluster_bundle: dict[str, Any] | None = None


def load_clustering_bundle():
    global cluster_bundle

    if not CLUSTER_BUNDLE_PATH.exists():
        print(
            f"Clustering bundle missing: {CLUSTER_BUNDLE_PATH}. "
            "Clustering endpoints will return HTTP 503."
        )
        cluster_bundle = None
        return

    loaded = joblib.load(CLUSTER_BUNDLE_PATH)
    if not isinstance(loaded, dict):
        raise ValueError("clustering_bundle.pkl must contain a dictionary.")

    required = {
        "model",
        "scaler",
        "feature_cols",
        "num_cols",
        "binary_cat_cols",
        "multi_cat_cols",
        "binary_encoders",
        "ordinal_encoder",
        "numeric_medians",
        "iqr_bounds",
    }
    missing = required - set(loaded)
    if missing:
        raise ValueError(
            "Clustering bundle missing keys: " + ", ".join(sorted(missing))
        )

    cluster_bundle = loaded
    print("Clustering bundle loaded; no training dataset was loaded.")


@app.on_event("startup")
def startup_event():
    try:
        load_clustering_bundle()
    except Exception as exc:
        print(f"Clustering bundle loading failed: {exc}")


def require_cluster_bundle() -> dict[str, Any]:
    if cluster_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Clustering model is not available. Create and deploy "
                "app/Pkl.files/clustering_bundle.pkl first."
            ),
        )
    return cluster_bundle


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Farmer Machine Learning API is running",
        "status": "success",
    }


@app.get("/health")
def health():
    regression_loaded = all(
        x is not None
        for x in (regression_model, regression_scaler, regression_encoder)
    )
    clustering_loaded = cluster_bundle is not None
    return {
        "status": "healthy" if regression_loaded and clustering_loaded else "degraded",
        "regression_model_loaded": regression_loaded,
        "clustering_model_loaded": clustering_loaded,
    }


# ============================================================
# CLUSTER FEATURE METADATA
# ============================================================

@app.get("/cluster/features")
def get_cluster_features():
    bundle = require_cluster_bundle()
    return {
        "status": "success",
        "total_features": len(bundle["feature_cols"]),
        "numerical_features": bundle["num_cols"],
        "binary_categorical_features": bundle["binary_cat_cols"],
        "multi_class_categorical_features": bundle["multi_cat_cols"],
        "all_features": bundle["feature_cols"],
    }


# ============================================================
# CLUSTER A SINGLE FARMER
# ============================================================

@app.post("/cluster")
def cluster_farmer(data: dict):
    bundle = require_cluster_bundle()

    try:
        row = pd.DataFrame([data])
        feature_cols = bundle["feature_cols"]
        num_cols = bundle["num_cols"]
        binary_cols = bundle["binary_cat_cols"]
        multi_cols = bundle["multi_cat_cols"]

        # Add absent expected fields using training-time defaults.
        for col in feature_cols:
            if col not in row.columns or pd.isna(row[col].iloc[0]):
                if col in num_cols:
                    row[col] = bundle["numeric_medians"].get(col, 0)
                elif col in binary_cols:
                    encoder = bundle["binary_encoders"][col]
                    row[col] = encoder.classes_[0]
                elif col in multi_cols:
                    # Use training-time mode saved in the bundle.
                    row[col] = bundle.get("categorical_modes", {}).get(
                        col, "Unknown"
                    )

        row = row.reindex(columns=feature_cols).copy()

        # Numeric conversion, imputation and training-time IQR caps.
        for col in num_cols:
            row[col] = pd.to_numeric(row[col], errors="coerce")
            row[col] = row[col].fillna(
                bundle["numeric_medians"].get(col, 0)
            )
            lower, upper = bundle["iqr_bounds"][col]
            row[col] = np.clip(row[col], lower, upper)

        # Binary encoders (same fitted encoders used during training).
        for col in binary_cols:
            encoder = bundle["binary_encoders"][col]
            value = row[col].iloc[0]
            if value not in encoder.classes_:
                value = encoder.classes_[0]
            row[col] = encoder.transform([value])

        # Ordinal encoder for multi-class categoricals.
        if multi_cols:
            encoder = bundle["ordinal_encoder"]
            for idx, col in enumerate(multi_cols):
                value = row[col].iloc[0]
                allowed = encoder.categories_[idx]
                if value not in allowed:
                    value = allowed[0]
                row[col] = value
            row[multi_cols] = encoder.transform(row[multi_cols])

        scaled = bundle["scaler"].transform(row[feature_cols])
        cluster_id = int(bundle["model"].predict(scaled)[0])

        return {
            "status": "success",
            "cluster": cluster_id,
            "farmer_segment": f"Group_{cluster_id}",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Clustering failed: {exc}"
        )


# ============================================================
# PRECOMPUTED SUMMARY ENDPOINTS
# These values are created offline; no full CSV is read at startup.
# ============================================================

def load_summary() -> dict[str, Any]:
    if not CLUSTER_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Precomputed cluster_summary.json is missing. "
                "Generate it offline and deploy it with the app."
            ),
        )
    import json
    with CLUSTER_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/cluster/summary")
def cluster_summary():
    summary = load_summary()
    counts = summary.get("cluster_counts", {})
    return {
        "status": "success",
        "number_of_clusters": len(counts) if counts else 4,
        "total_farmers": int(summary.get("total_farmers", sum(counts.values()))),
        "cluster_counts": counts,
    }


@app.get("/segments")
def get_segments():
    summary = load_summary()
    counts = summary.get("cluster_counts", {})
    return {
        "status": "success",
        "segments": {
            f"Group_{key}": int(value) for key, value in counts.items()
        },
    }


@app.get("/cluster/statistics")
def cluster_statistics():
    summary = load_summary()
    return {
        "status": "success",
        "statistics": summary.get("statistics", []),
    }

