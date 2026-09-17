
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

DATA_PATH = PROJECT_DIR / "Data" / "clustered_farmers_data.csv"
ARTIFACT_DIR = PROJECT_DIR / "app" / "Pkl.files"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# REMOVE TARGET / IDENTIFIER COLUMNS
# --------------------------------------------------

exclude_cols = [
    "record_id",
    "household_id",
    "Cluster",
    "Farmer_Segment",
]

X = df.drop(
    columns=exclude_cols,
    errors="ignore",
).copy()

# Drop columns that contain no useful values.
X = X.dropna(axis=1, how="all")


# --------------------------------------------------
# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# --------------------------------------------------

num_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

cat_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# Convert categorical values to strings and fill missing.
for col in cat_cols:
    X[col] = X[col].fillna("Unknown").astype(str)

# Fill missing numeric values with medians.
numeric_medians = {}

for col in num_cols:
    median = X[col].median()

    if pd.isna(median):
        median = 0

    numeric_medians[col] = float(median)

    X[col] = X[col].fillna(median)


# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

ordinal_encoder = None

if cat_cols:
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X[cat_cols] = ordinal_encoder.fit_transform(
        X[cat_cols]
    )


# --------------------------------------------------
# SCALE FEATURES
# --------------------------------------------------

feature_cols = X.columns.tolist()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X[feature_cols])


# --------------------------------------------------
# TRAIN K-MEANS
# --------------------------------------------------

print("Training K-Means...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
)

cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels
df["Farmer_Segment"] = [
    f"Group_{label}" for label in cluster_labels
]


# --------------------------------------------------
# SAVE CLUSTERING BUNDLE
# --------------------------------------------------

bundle = {
    "model": kmeans,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "num_cols": num_cols,
    "binary_cat_cols": [],
    "multi_cat_cols": cat_cols,
    "binary_encoders": {},
    "ordinal_encoder": ordinal_encoder,
    "numeric_medians": numeric_medians,
    "iqr_bounds": {
        col: (
            float(X[col].min()),
            float(X[col].max()),
        )
        for col in num_cols
    },
    "categorical_modes": {
        col: str(df[col].mode().iloc[0])
        for col in cat_cols
        if not df[col].mode().empty
    },
}

bundle_path = ARTIFACT_DIR / "clustering_bundle.pkl"

joblib.dump(bundle, bundle_path)

print("Saved:", bundle_path)


# --------------------------------------------------
# SAVE SUMMARY
# --------------------------------------------------

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .to_dict()
)

cluster_counts = {
    str(k): int(v)
    for k, v in cluster_counts.items()
}

summary = {
    "total_farmers": int(len(df)),
    "cluster_counts": cluster_counts,
    "statistics": [],
}

summary_path = ARTIFACT_DIR / "cluster_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved:", summary_path)

print("Clustering artifact generation completed.")