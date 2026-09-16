from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Farmer Machine Learning API",
    description="Farmer Income Prediction and Farmer Clustering API",
    version="1.0.0"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

APP_DIR = BASE_DIR.parent

PROJECT_DIR = BASE_DIR.parent.parent

MODEL_DIR = APP_DIR / "Pkl.files"

DATA_DIR = PROJECT_DIR / "Data"

DATA_PATH = DATA_DIR / "agricultural_productivity_Cleaned.csv"

OUTPUT_PATH = DATA_DIR / "clustered_farmers_data.csv"


# ============================================================
# REGRESSION MODEL PATHS
# ============================================================

MODEL_PATH = MODEL_DIR / "best_regression_model.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

ENCODER_PATH = MODEL_DIR / "encoder.pkl"


# ============================================================
# LOAD REGRESSION MODEL
# ============================================================

try:

    regression_model = joblib.load(MODEL_PATH)

    regression_scaler = joblib.load(SCALER_PATH)

    regression_encoder = joblib.load(ENCODER_PATH)

    print("======================================")
    print("Regression model loaded successfully")
    print("Regression scaler loaded successfully")
    print("Regression encoder loaded successfully")
    print("======================================")

except Exception as e:

    print("ERROR loading regression artifacts:")
    print(e)

    regression_model = None
    regression_scaler = None
    regression_encoder = None


# ============================================================
# REGRESSION INPUT
# ============================================================

class PredictionInput(BaseModel):

    farming_system: str = Field(
        ...,
        description="Farming system used by the farmer"
    )

    head_of_household_age: float = Field(
        ...,
        ge=0,
        description="Age of head of household"
    )

    land_owned_hectares: float = Field(
        ...,
        ge=0,
        description="Land owned in hectares"
    )

    fertilizer_used_kg_per_hectare: float = Field(
        ...,
        ge=0,
        description="Fertilizer used per hectare"
    )

    goats_number: float = Field(
        ...,
        ge=0,
        description="Number of goats"
    )

    sheep_number: float = Field(
        ...,
        ge=0,
        description="Number of sheep"
    )

    livestock_eggs_per_week: float = Field(
        ...,
        ge=0,
        description="Livestock eggs produced per week"
    )

    livestock_milk_litres_per_week: float = Field(
        ...,
        ge=0,
        description="Livestock milk produced per week"
    )


# ============================================================
# REGRESSION FEATURE ENGINEERING
# ============================================================

def create_regression_features(df_input):

    df_feat = df_input.copy()

    df_feat["total_livestock_count"] = (
        df_feat["goats_number"]
        +
        df_feat["sheep_number"]
    )

    df_feat["fertilizer_intensity"] = (
        df_feat["fertilizer_used_kg_per_hectare"]
        /
        (
            df_feat["land_owned_hectares"]
            +
            0.01
        )
    )

    df_feat["weekly_animal_yield"] = (
        df_feat["livestock_eggs_per_week"]
        +
        df_feat["livestock_milk_litres_per_week"]
    )

    df_feat["is_senior_farmer"] = np.where(
        df_feat["head_of_household_age"] >= 60,
        1,
        0
    )

    return df_feat


# ============================================================
# REGRESSION FEATURES
# ============================================================

regression_cat_cols = [
    "farming_system"
]

regression_num_cols = [

    "head_of_household_age",

    "land_owned_hectares",

    "fertilizer_used_kg_per_hectare",

    "goats_number",

    "sheep_number",

    "livestock_eggs_per_week",

    "livestock_milk_litres_per_week"

]

regression_new_num_cols = (

    regression_num_cols

    +

    [
        "total_livestock_count",
        "fertilizer_intensity",
        "weekly_animal_yield",
        "is_senior_farmer"
    ]

)

regression_skewed_cols = [

    "land_owned_hectares",

    "fertilizer_used_kg_per_hectare",

    "fertilizer_intensity"

]


# ============================================================
# REGRESSION PREDICTION
# ============================================================

@app.post("/predict")
def predict_income(data: PredictionInput):

    if (
        regression_model is None
        or regression_scaler is None
        or regression_encoder is None
    ):

        raise HTTPException(
            status_code=500,
            detail="Regression model is not loaded"
        )

    try:

        input_data = pd.DataFrame(
            [data.model_dump()]
        )

        input_fe = create_regression_features(
            input_data
        )

        for col in regression_skewed_cols:

            input_fe[col] = np.log1p(
                np.maximum(
                    0,
                    input_fe[col]
                )
            )

        encoded_cat = regression_encoder.transform(
            input_fe[regression_cat_cols]
        )

        encoded_cat_cols = (
            regression_encoder
            .get_feature_names_out(
                regression_cat_cols
            )
        )

        encoded_cat_df = pd.DataFrame(
            encoded_cat,
            columns=encoded_cat_cols,
            index=input_fe.index
        )

        input_prep = pd.concat(
            [
                input_fe[
                    regression_new_num_cols
                ],

                encoded_cat_df
            ],
            axis=1
        )

        input_scaled = regression_scaler.transform(
            input_prep
        )

        predicted_log_income = (
            regression_model.predict(
                input_scaled
            )
        )

        predicted_income = np.expm1(
            predicted_log_income
        )

        return {
            "status": "success",
            "predicted_income_ngn": round(
                float(predicted_income[0]),
                2
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# CLUSTERING GLOBAL VARIABLES
# ============================================================

clustering_df = None

clustering_model = None

clustering_scaler = None

clustering_binary_encoders = {}

clustering_ordinal_encoder = None

cluster_num_cols = []

cluster_cat_cols = []

cluster_binary_cat_cols = []

cluster_multi_cat_cols = []

cluster_feature_cols = []

cluster_numeric_medians = {}

cluster_iqr_bounds = {}


# ============================================================
# LOAD CLUSTERING DATASET
# ============================================================

def load_clustering_dataset():

    global clustering_df

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    clustering_df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Clustering dataset loaded: "
        f"{clustering_df.shape[0]:,} rows, "
        f"{clustering_df.shape[1]} columns"
    )


# ============================================================
# TRAIN CLUSTERING MODEL
# ============================================================

def train_clustering_model():

    global clustering_model
    global clustering_scaler

    global clustering_binary_encoders
    global clustering_ordinal_encoder

    global cluster_num_cols
    global cluster_cat_cols

    global cluster_binary_cat_cols
    global cluster_multi_cat_cols

    global cluster_feature_cols

    global cluster_numeric_medians
    global cluster_iqr_bounds


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    load_clustering_dataset()


    # --------------------------------------------------------
    # Create X
    # --------------------------------------------------------

    X = clustering_df.copy()


    # --------------------------------------------------------
    # Remove ID columns
    # --------------------------------------------------------

    X = X.drop(
        columns=[
            "record_id",
            "household_id"
        ],
        errors="ignore"
    )


    # --------------------------------------------------------
    # Detect numerical columns
    # --------------------------------------------------------

    cluster_num_cols = (
        X.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


    # --------------------------------------------------------
    # Detect categorical columns
    # --------------------------------------------------------

    cluster_cat_cols = (
        X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )


    print("Numerical columns:")
    print(cluster_num_cols)

    print("Categorical columns:")
    print(cluster_cat_cols)


    # --------------------------------------------------------
    # Numerical missing values
    # --------------------------------------------------------

    cluster_numeric_medians = {}

    for col in cluster_num_cols:

        median_value = X[col].median()

        cluster_numeric_medians[col] = median_value

        X[col] = X[col].fillna(
            median_value
        )


    # --------------------------------------------------------
    # Categorical missing values
    # --------------------------------------------------------

    for col in cluster_cat_cols:

        if X[col].isnull().sum() > 0:

            mode_values = X[col].mode()

            if len(mode_values) > 0:

                X[col] = X[col].fillna(
                    mode_values.iloc[0]
                )


    # --------------------------------------------------------
    # IQR outlier capping
    # --------------------------------------------------------

    cluster_iqr_bounds = {}

    for col in cluster_num_cols:

        Q1 = X[col].quantile(0.25)

        Q3 = X[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR

        cluster_iqr_bounds[col] = (
            lower_bound,
            upper_bound
        )

        X[col] = np.clip(
            X[col],
            lower_bound,
            upper_bound
        )


    # --------------------------------------------------------
    # Binary categorical columns
    # --------------------------------------------------------

    cluster_binary_cat_cols = [

        col

        for col in cluster_cat_cols

        if X[col].nunique() == 2

    ]


    # --------------------------------------------------------
    # Multi-class categorical columns
    # --------------------------------------------------------

    cluster_multi_cat_cols = [

        col

        for col in cluster_cat_cols

        if X[col].nunique() > 2

    ]


    # --------------------------------------------------------
    # Encoding
    # --------------------------------------------------------

    X_encoded = X.copy()

    clustering_binary_encoders = {}


    # Binary encoding

    for col in cluster_binary_cat_cols:

        encoder = LabelEncoder()

        X_encoded[col] = encoder.fit_transform(
            X_encoded[col]
        )

        clustering_binary_encoders[col] = (
            encoder
        )


    # Multi-class encoding

    clustering_ordinal_encoder = None

    if len(cluster_multi_cat_cols) > 0:

        clustering_ordinal_encoder = (
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )

        X_encoded[
            cluster_multi_cat_cols
        ] = (
            clustering_ordinal_encoder.fit_transform(
                X_encoded[
                    cluster_multi_cat_cols
                ]
            )
        )


    # --------------------------------------------------------
    # Final feature list
    # --------------------------------------------------------

    cluster_feature_cols = (

        cluster_num_cols

        +

        cluster_binary_cat_cols

        +

        cluster_multi_cat_cols

    )


    print("======================================")
    print("FINAL CLUSTERING FEATURES")
    print("======================================")

    for feature in cluster_feature_cols:

        print(feature)

    print("======================================")


    X_selected = X_encoded[
        cluster_feature_cols
    ]


    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    clustering_scaler = StandardScaler()

    X_scaled = clustering_scaler.fit_transform(
        X_selected
    )


    # --------------------------------------------------------
    # K-Means
    # --------------------------------------------------------

    clustering_model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )


    cluster_numbers = (
        clustering_model.fit_predict(
            X_scaled
        )
    )


    # --------------------------------------------------------
    # Add cluster
    # --------------------------------------------------------

    clustering_df["Cluster"] = (
        cluster_numbers
    )

    clustering_df["Farmer_Segment"] = [

        f"Group_{cluster}"

        for cluster in cluster_numbers

    ]


    # --------------------------------------------------------
    # Save clustered dataset
    # --------------------------------------------------------

    clustering_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "K-Means clustering completed successfully"
    )

    print("Cluster counts:")

    print(
        clustering_df[
            "Cluster"
        ]
        .value_counts()
        .sort_index()
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    try:

        train_clustering_model()

        print(
            "Clustering model initialized successfully"
        )

    except Exception as e:

        print(
            "Clustering initialization failed:"
        )

        print(e)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Farmer Machine Learning API is running",

        "status":
            "success"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    regression_loaded = (

        regression_model is not None

        and

        regression_scaler is not None

        and

        regression_encoder is not None

    )


    clustering_loaded = (

        clustering_model is not None

        and

        clustering_scaler is not None

    )


    return {

        "status":
            "healthy"

            if (
                regression_loaded
                and clustering_loaded
            )

            else "unhealthy",

        "regression_model_loaded":
            regression_loaded,

        "clustering_model_loaded":
            clustering_loaded

    }


# ============================================================
# CLUSTER FEATURES
# ============================================================

@app.get("/cluster/features")
def get_cluster_features():

    if not cluster_feature_cols:

        raise HTTPException(
            status_code=500,
            detail="Clustering model is not initialized"
        )


    return {

        "status":
            "success",

        "total_features":
            len(cluster_feature_cols),

        "numerical_features":
            cluster_num_cols,

        "binary_categorical_features":
            cluster_binary_cat_cols,

        "multi_class_categorical_features":
            cluster_multi_cat_cols,

        "all_features":
            cluster_feature_cols

    }


# ============================================================
# POST /CLUSTER
# ============================================================

@app.post("/cluster")
def cluster_farmer(data: dict):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if (

        clustering_model is None

        or

        clustering_scaler is None

    ):

        raise HTTPException(

            status_code=500,

            detail=
                "Clustering model is not loaded"

        )


    try:

        # ----------------------------------------------------
        # JSON → DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [data]
        )


        # ----------------------------------------------------
        # Auto-fill missing features with default values
        # ----------------------------------------------------

        for col in cluster_feature_cols:
            if col not in input_df.columns or pd.isna(input_df[col].iloc[0]):
                if col in cluster_num_cols:
                    input_df[col] = cluster_numeric_medians.get(col, 0)
                elif col in cluster_binary_cat_cols:
                    input_df[col] = clustering_binary_encoders[col].classes_[0]
                elif col in cluster_multi_cat_cols:
                    if clustering_df is not None and col in clustering_df.columns:
                        input_df[col] = clustering_df[col].mode()[0]
                    else:
                        input_df[col] = "Unknown"

        # ----------------------------------------------------
        # Select correct feature order
        # ----------------------------------------------------

        input_df = input_df[
            cluster_feature_cols
        ].copy()


        # ----------------------------------------------------
        # Numerical processing
        # ----------------------------------------------------

        for col in cluster_num_cols:

            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="coerce"
            )

            input_df[col] = input_df[col].fillna(
                cluster_numeric_medians[col]
            )


        # ----------------------------------------------------
        # IQR capping
        # ----------------------------------------------------

        for col in cluster_num_cols:

            lower_bound, upper_bound = (
                cluster_iqr_bounds[col]
            )

            input_df[col] = np.clip(
                input_df[col],
                lower_bound,
                upper_bound
            )


        # ----------------------------------------------------
        # Binary encoding
        # ----------------------------------------------------

        for col in cluster_binary_cat_cols:

            encoder = (
                clustering_binary_encoders[col]
            )

            value = input_df[col].iloc[0]


            if value not in encoder.classes_:
                input_df[col] = encoder.classes_[0]


            input_df[col] = encoder.transform(
                input_df[col]
            )


        # ----------------------------------------------------
        # Multi-class encoding
        # ----------------------------------------------------

        if len(cluster_multi_cat_cols) > 0:
            for idx, col in enumerate(cluster_multi_cat_cols):
                val = input_df[col].iloc[0]
                allowed = clustering_ordinal_encoder.categories_[idx]
                if val not in allowed:
                    input_df[col] = allowed[0]

            input_df[
                cluster_multi_cat_cols
            ] = (
                clustering_ordinal_encoder.transform(
                    input_df[
                        cluster_multi_cat_cols
                    ]
                )
            )


        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        input_scaled = (
            clustering_scaler.transform(
                input_df[
                    cluster_feature_cols
                ]
            )
        )


        # ----------------------------------------------------
        # Predict cluster
        # ----------------------------------------------------

        cluster_id = (
            clustering_model.predict(
                input_scaled
            )[0]
        )


        # ----------------------------------------------------
        # Farmer segment
        # ----------------------------------------------------

        farmer_segment = (
            f"Group_{int(cluster_id)}"
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "status":
                "success",

            "cluster":
                int(cluster_id),

            "farmer_segment":
                farmer_segment

        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Clustering failed: {str(e)}"

        )


# ============================================================
# CLUSTER SUMMARY
# ============================================================

@app.get("/cluster/summary")
def cluster_summary():

    if clustering_df is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Clustering dataset is not loaded"

        )


    counts = (

        clustering_df[
            "Cluster"
        ]

        .value_counts()

        .sort_index()

        .to_dict()

    )


    return {

        "status":
            "success",

        "number_of_clusters":
            4,

        "total_farmers":
            int(
                len(clustering_df)
            ),

        "cluster_counts": {

            str(key):
                int(value)

            for key, value
            in counts.items()

        }

    }


# ============================================================
# FARMER SEGMENTS
# ============================================================

@app.get("/segments")
def get_segments():

    if clustering_df is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Clustering dataset is not loaded"

        )


    segments = (

        clustering_df[
            "Farmer_Segment"
        ]

        .value_counts()

        .sort_index()

        .to_dict()

    )


    return {

        "status":
            "success",

        "segments": {

            str(key):
                int(value)

            for key, value
            in segments.items()

        }

    }


# ============================================================
# CLUSTER STATISTICS
# ============================================================

@app.get("/cluster/statistics")
def cluster_statistics():

    if clustering_df is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Clustering dataset is not loaded"

        )


    key_indicators = [

        "land_owned_hectares",

        "fertilizer_used_kg_per_hectare",

        "total_household_income_ngn",

        "agricultural_income_ngn",

        "market_access_distance_km",

        "food_consumption_score"

    ]


    available_indicators = [

        col

        for col in key_indicators

        if col in clustering_df.columns

    ]


    if not available_indicators:

        return {

            "status":
                "success",

            "statistics":
                []

        }


    stats = (

        clustering_df

        .groupby(
            "Farmer_Segment"
        )[

            available_indicators

        ]

        .mean()

        .round(2)

    )


    return {

        "status":
            "success",

        "statistics":

            stats

            .reset_index()

            .to_dict(
                orient="records"
            )

    }
@app.get("/cluster/features")
def get_cluster_features():
    if not cluster_feature_cols:
        raise HTTPException(
            status_code=500,
            detail="Clustering model is not initialized"
        )

    return {
        "status": "success",
        "total_features": len(cluster_feature_cols),
        "numerical_features": cluster_num_cols,
        "binary_categorical_features": cluster_binary_cat_cols,
        "multi_class_categorical_features": cluster_multi_cat_cols,
        "all_features": cluster_feature_cols
    }