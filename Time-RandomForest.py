import pm4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD XES FILE
# ============================================================

log = pm4py.read_xes("Event Log.xes")
df = pm4py.convert_to_dataframe(log)

print("Loaded:", df.shape)

df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

df = df.sort_values(
    ["case:concept:name", "time:timestamp"]
).reset_index(drop=True)


# ============================================================
# 2. CALCULATE FINAL CASE DURATION
# ============================================================

case_summary = (
    df.groupby("case:concept:name")["time:timestamp"]
    .agg(
        case_start="min",
        case_end="max"
    )
    .reset_index()
)

case_summary["case_duration_days"] = (
    case_summary["case_end"]
    - case_summary["case_start"]
).dt.total_seconds() / 86400


# ============================================================
# 3. DEFINE TARGET
# ============================================================

LATE_THRESHOLD = 102.38

case_summary["late"] = (
    case_summary["case_duration_days"] > LATE_THRESHOLD
).astype(int)

print("\n=== TARGET DISTRIBUTION ===")
print(case_summary["late"].value_counts())

print(
    case_summary["late"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 4. KEEP ONLY CASES WITH AT LEAST 2 EVENTS
# ============================================================

event_counts = (
    df.groupby("case:concept:name")
    .size()
)

valid_cases = event_counts[
    event_counts >= 2
].index

df_valid = df[
    df["case:concept:name"].isin(valid_cases)
].copy()

print(
    "\nCases with at least 2 events:",
    len(valid_cases)
)


# ============================================================
# 5. CREATE FEATURES AFTER FIRST 2 EVENTS
# ============================================================

rows = []

for case_id, group in df_valid.groupby("case:concept:name"):

    group = group.sort_values("time:timestamp")

    prefix = group.iloc[:2]

    event_1 = prefix.iloc[0]
    event_2 = prefix.iloc[1]

    # Time from first to second event
    elapsed_time_to_2nd_event = (
        event_2["time:timestamp"]
        - event_1["time:timestamp"]
    ).total_seconds() / 86400

    # Whether CHANGE DIAGN occurred
    change_diagn_so_far = int(
        (prefix["concept:name"] == "CHANGE DIAGN").any()
    )

    # Case-level attributes
    diagnosis = event_1.get("diagnosis", np.nan)
    case_type = event_1.get("caseType", np.nan)
    speciality = event_1.get("speciality", np.nan)
    blocked = event_1.get("blocked", np.nan)

    # Latest available resource within first 2 events
    resource_values = (
        prefix["org:resource"]
        .dropna()
        .tolist()
    )

    if len(resource_values) > 0:
        org_resource = resource_values[-1]
    else:
        org_resource = np.nan

    rows.append({

        "case:concept:name": case_id,

        "activity_1":
            event_1["concept:name"],

        "activity_2":
            event_2["concept:name"],

        "elapsed_time_to_2nd_event":
            elapsed_time_to_2nd_event,

        "change_diagn_so_far":
            change_diagn_so_far,

        "diagnosis":
            diagnosis,

        "caseType":
            case_type,

        "speciality":
            speciality,

        "org_resource":
            org_resource,

        "blocked":
            blocked
    })


features_df = pd.DataFrame(rows)


# ============================================================
# 6. MERGE FEATURES WITH TARGET
# ============================================================

model_df = features_df.merge(

    case_summary[
        [
            "case:concept:name",
            "case_start",
            "case_duration_days",
            "late"
        ]
    ],

    on="case:concept:name",
    how="inner"
)

print("\n=== MODEL DATASET ===")
print("Shape:", model_df.shape)

print("\nLate distribution:")
print(model_df["late"].value_counts())


# ============================================================
# 7. CHRONOLOGICAL 80/20 TRAIN-TEST SPLIT
# ============================================================

model_df = model_df.sort_values(
    "case_start"
).reset_index(drop=True)

split_index = int(
    len(model_df) * 0.80
)

train_df = model_df.iloc[
    :split_index
].copy()

test_df = model_df.iloc[
    split_index:
].copy()

print("\n=== TRAIN / TEST SPLIT ===")

print("Training cases:", len(train_df))
print("Testing cases:", len(test_df))

print(
    "Training percentage:",
    round(
        len(train_df)
        / len(model_df)
        * 100,
        2
    ),
    "%"
)

print(
    "Testing percentage:",
    round(
        len(test_df)
        / len(model_df)
        * 100,
        2
    ),
    "%"
)


# ============================================================
# 8. DEFINE FEATURES
# ============================================================

feature_columns = [
    "activity_1",
    "activity_2",
    "elapsed_time_to_2nd_event",
    "change_diagn_so_far",
    "diagnosis",
    "caseType",
    "speciality",
    "org_resource",
    "blocked"
]

X_train = train_df[feature_columns]
y_train = train_df["late"]

X_test = test_df[feature_columns]
y_test = test_df["late"]


# ============================================================
# 9. FEATURE TYPES
# ============================================================

numeric_features = [
    "elapsed_time_to_2nd_event",
    "change_diagn_so_far"
]

categorical_features = [
    "activity_1",
    "activity_2",
    "diagnosis",
    "caseType",
    "speciality",
    "org_resource",
    "blocked"
]


# ============================================================
# 10. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numeric",

            "passthrough",

            numeric_features
        )
    ]
)


# ============================================================
# 11. RANDOM FOREST MODEL
# ============================================================

rf_classifier = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_leaf=20,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            rf_classifier
        )
    ]
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 13. PREDICT
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_probability
)

print(
    "\n=== RANDOM FOREST RESULTS ==="
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

print(
    f"AUC      : {auc:.4f}"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print(
    "\n=== CLASSIFICATION REPORT ==="
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "On-time",
            "Late"
        ]
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print(
    "\n=== CONFUSION MATRIX ==="
)

print(cm)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

fitted_preprocessor = (
    model.named_steps[
        "preprocessor"
    ]
)

encoded_feature_names = (
    fitted_preprocessor
    .named_transformers_[
        "categorical"
    ]
    .get_feature_names_out(
        categorical_features
    )
)

all_feature_names = (
    list(encoded_feature_names)
    + numeric_features
)

feature_importance = (
    model.named_steps[
        "classifier"
    ]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": feature_importance
}).sort_values(
    "Importance",
    ascending=False
)

print(
    "\n=== TOP 20 RANDOM FOREST FEATURE IMPORTANCES ==="
)

print(
    importance_df.head(20)
)


# ============================================================
# 18. PLOT FEATURE IMPORTANCE
# ============================================================

top_features = (
    importance_df
    .head(15)
    .sort_values(
        "Importance"
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest - Top Feature Importances"
)

plt.tight_layout()

plt.show()