import pm4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
# 2. DEFINE COMPLIANCE TARGET
#
# 0 = Compliant
# 1 = Non-Compliant
# ============================================================

def classify_compliance(group):

    group = group.sort_values(
        "time:timestamp"
    ).reset_index(drop=True)

    activities = group["concept:name"].tolist()


    # --------------------------------------------------------
    # Exclude cases that never reached BILLED
    # --------------------------------------------------------

    if "BILLED" not in activities:
        return np.nan


    # --------------------------------------------------------
    # Exclude cancelled cases
    # --------------------------------------------------------

    if "isCancelled" in group.columns:

        cancelled_values = (
            group["isCancelled"]
            .dropna()
            .astype(str)
            .str.lower()
        )

        if cancelled_values.isin(
            ["true", "1", "yes"]
        ).any():
            return np.nan


    # --------------------------------------------------------
    # First BILLED event
    # --------------------------------------------------------

    billed_index = activities.index("BILLED")

    before_billing = activities[:billed_index]


    # --------------------------------------------------------
    # Find FIN
    # --------------------------------------------------------

    fin_positions = [
        i for i, activity
        in enumerate(before_billing)
        if activity == "FIN"
    ]


    # --------------------------------------------------------
    # Find RELEASE
    # --------------------------------------------------------

    release_positions = [
        i for i, activity
        in enumerate(before_billing)
        if activity == "RELEASE"
    ]


    # --------------------------------------------------------
    # Find coding validation events
    # --------------------------------------------------------

    coding_activities = {
        "CODE OK",
        "CODE NOK",
        "CODE ERROR"
    }

    coding_positions = [
        (i, activity)
        for i, activity
        in enumerate(before_billing)
        if activity in coding_activities
    ]


    # --------------------------------------------------------
    # Missing required controls = Non-Compliant
    # --------------------------------------------------------

    if len(fin_positions) == 0:
        return 1

    if len(release_positions) == 0:
        return 1

    if len(coding_positions) == 0:
        return 1


    # --------------------------------------------------------
    # Latest relevant controls before billing
    # --------------------------------------------------------

    last_fin = max(fin_positions)

    last_release = max(release_positions)

    last_code_position, last_code_status = (
        coding_positions[-1]
    )


    # --------------------------------------------------------
    # Required sequence:
    #
    # FIN -> RELEASE -> CODE OK -> BILLED
    # --------------------------------------------------------

    correct_order = (
        last_fin
        < last_release
        < last_code_position
        < billed_index
    )

    successful_code = (
        last_code_status == "CODE OK"
    )


    if correct_order and successful_code:

        # Compliant
        return 0

    else:

        # Non-Compliant
        return 1


# ============================================================
# 3. CREATE TARGET FOR EACH CASE
# ============================================================

compliance_rows = []

for case_id, group in df.groupby(
    "case:concept:name"
):

    group = group.sort_values(
        "time:timestamp"
    )

    compliance_rows.append({

        "case:concept:name":
            case_id,

        "case_start":
            group["time:timestamp"].min(),

        "non_compliant":
            classify_compliance(group)
    })


case_summary = pd.DataFrame(
    compliance_rows
)


# Remove cases where compliance cannot be evaluated

case_summary = (
    case_summary
    .dropna(subset=["non_compliant"])
    .copy()
)

case_summary["non_compliant"] = (
    case_summary["non_compliant"]
    .astype(int)
)


print(
    "\n=== COMPLIANCE TARGET DISTRIBUTION ==="
)

print(
    case_summary[
        "non_compliant"
    ].value_counts()
)

print(
    case_summary[
        "non_compliant"
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 4. KEEP CASES WITH AT LEAST 2 EVENTS
# ============================================================

event_counts = (
    df.groupby("case:concept:name")
    .size()
)

cases_with_two_events = (
    event_counts[
        event_counts >= 2
    ].index
)


valid_case_ids = (
    case_summary[
        case_summary["case:concept:name"]
        .isin(cases_with_two_events)
    ]["case:concept:name"]
)


df_valid = df[
    df["case:concept:name"]
    .isin(valid_case_ids)
].copy()


print(
    "\nCases available for compliance prediction:",
    len(valid_case_ids)
)


# ============================================================
# 5. CREATE FEATURES FROM FIRST TWO EVENTS
# ============================================================

rows = []

for case_id, group in df_valid.groupby(
    "case:concept:name"
):

    group = group.sort_values(
        "time:timestamp"
    )

    prefix = group.iloc[:2]

    event_1 = prefix.iloc[0]
    event_2 = prefix.iloc[1]


    elapsed_time_to_2nd_event = (
        event_2["time:timestamp"]
        - event_1["time:timestamp"]
    ).total_seconds() / 86400


    change_diagn_so_far = int(
        (
            prefix["concept:name"]
            == "CHANGE DIAGN"
        ).any()
    )


    diagnosis = event_1.get(
        "diagnosis",
        np.nan
    )

    case_type = event_1.get(
        "caseType",
        np.nan
    )

    speciality = event_1.get(
        "speciality",
        np.nan
    )

    blocked = event_1.get(
        "blocked",
        np.nan
    )


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

        "case:concept:name":
            case_id,

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
            "non_compliant"
        ]
    ],

    on="case:concept:name",
    how="inner"
)


print("\n=== MODEL DATASET ===")

print(
    "Shape:",
    model_df.shape
)

print(
    "\nCompliance distribution:"
)

print(
    model_df[
        "non_compliant"
    ].value_counts()
)


# ============================================================
# 7. CHRONOLOGICAL 80/20 SPLIT
# ============================================================

model_df = (
    model_df
    .sort_values("case_start")
    .reset_index(drop=True)
)

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

print(
    "Training cases:",
    len(train_df)
)

print(
    "Testing cases:",
    len(test_df)
)


# ============================================================
# 8. FEATURES
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


X_train = train_df[
    feature_columns
]

y_train = train_df[
    "non_compliant"
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    "non_compliant"
]


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

from xgboost import XGBClassifier


# ============================================================
# CLASS WEIGHT
# ============================================================

negative_cases = (
    y_train == 0
).sum()

positive_cases = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_cases
    / positive_cases
)


print(
    "\nscale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# XGBOOST MODEL
# ============================================================

xgb_classifier = XGBClassifier(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=
        scale_pos_weight,

    objective=
        "binary:logistic",

    eval_metric=
        "logloss",

    random_state=42,

    n_jobs=-1
)


xgb_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            xgb_classifier
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

xgb_model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICT
# ============================================================

y_pred_xgb = xgb_model.predict(
    X_test
)

y_prob_xgb = (
    xgb_model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# EVALUATION
# ============================================================

print(
    "\n=== XGBOOST COMPLIANCE RESULTS ==="
)

print(
    f"Accuracy : "
    f"{accuracy_score(y_test, y_pred_xgb):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_test, y_pred_xgb, zero_division=0):.4f}"
)

print(
    f"Recall   : "
    f"{recall_score(y_test, y_pred_xgb, zero_division=0):.4f}"
)

print(
    f"F1-score : "
    f"{f1_score(y_test, y_pred_xgb, zero_division=0):.4f}"
)

print(
    f"AUC      : "
    f"{roc_auc_score(y_test, y_prob_xgb):.4f}"
)


print(
    "\n=== CLASSIFICATION REPORT ==="
)

print(
    classification_report(
        y_test,
        y_pred_xgb,
        target_names=[
            "Compliant",
            "Non-Compliant"
        ],
        zero_division=0
    )
)


print(
    "\n=== CONFUSION MATRIX ==="
)

print(
    confusion_matrix(
        y_test,
        y_pred_xgb
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

fitted_preprocessor_xgb = (
    xgb_model.named_steps[
        "preprocessor"
    ]
)


encoded_features_xgb = (
    fitted_preprocessor_xgb
    .named_transformers_[
        "categorical"
    ]
    .get_feature_names_out(
        categorical_features
    )
)


all_feature_names_xgb = (
    list(
        encoded_features_xgb
    )
    + numeric_features
)


xgb_importance = (
    xgb_model
    .named_steps[
        "classifier"
    ]
    .feature_importances_
)


xgb_importance_df = pd.DataFrame({

    "Feature":
        all_feature_names_xgb,

    "Importance":
        xgb_importance

}).sort_values(
    "Importance",
    ascending=False
)


print(
    "\n=== TOP 20 XGBOOST FEATURES ==="
)

print(
    xgb_importance_df.head(20)
)


# ============================================================
# PLOT TOP FEATURES
# ============================================================

top_xgb = (
    xgb_importance_df
    .head(15)
    .sort_values(
        "Importance"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_xgb["Feature"],
    top_xgb["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.title(
    "XGBoost - Compliance Prediction"
)

plt.tight_layout()

plt.show()

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)


# ============================================================
# DECISION TREE MODEL
# ============================================================

tree_classifier = DecisionTreeClassifier(

    max_depth=5,

    min_samples_leaf=50,

    class_weight="balanced",

    random_state=42
)


dt_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            tree_classifier
        )
    ]
)


# Train

dt_model.fit(
    X_train,
    y_train
)


# Predict

y_pred_dt = dt_model.predict(
    X_test
)

y_prob_dt = dt_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

print(
    "\n=== DECISION TREE COMPLIANCE RESULTS ==="
)

print(
    f"Accuracy : "
    f"{accuracy_score(y_test, y_pred_dt):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_test, y_pred_dt, zero_division=0):.4f}"
)

print(
    f"Recall   : "
    f"{recall_score(y_test, y_pred_dt, zero_division=0):.4f}"
)

print(
    f"F1-score : "
    f"{f1_score(y_test, y_pred_dt, zero_division=0):.4f}"
)

print(
    f"AUC      : "
    f"{roc_auc_score(y_test, y_prob_dt):.4f}"
)


print(
    "\n=== CLASSIFICATION REPORT ==="
)

print(
    classification_report(
        y_test,
        y_pred_dt,
        target_names=[
            "Compliant",
            "Non-Compliant"
        ],
        zero_division=0
    )
)


print(
    "\n=== CONFUSION MATRIX ==="
)

print(
    confusion_matrix(
        y_test,
        y_pred_dt
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

fitted_preprocessor = (
    dt_model.named_steps[
        "preprocessor"
    ]
)

encoded_features = (
    fitted_preprocessor
    .named_transformers_[
        "categorical"
    ]
    .get_feature_names_out(
        categorical_features
    )
)

all_feature_names = (
    list(encoded_features)
    + numeric_features
)


dt_importance = (
    dt_model
    .named_steps[
        "classifier"
    ]
    .feature_importances_
)


dt_importance_df = pd.DataFrame({

    "Feature":
        all_feature_names,

    "Importance":
        dt_importance

}).sort_values(
    "Importance",
    ascending=False
)


print(
    "\n=== TOP 20 DECISION TREE FEATURES ==="
)

print(
    dt_importance_df.head(20)
)


# ============================================================
# PLOT TOP FEATURES
# ============================================================

top_dt = (
    dt_importance_df
    .head(15)
    .sort_values("Importance")
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_dt["Feature"],
    top_dt["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.title(
    "Decision Tree - Compliance Prediction"
)

plt.tight_layout()

plt.show()