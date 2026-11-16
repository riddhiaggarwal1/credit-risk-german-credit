"""Credit risk analysis on the German Credit dataset (UCI / Statlog).

Steps
  1. Load the raw file and translate its codes (A11, A12, ...) into readable labels
  2. Explore: which applicant characteristics go together with default?
  3. Model: logistic regression (scorecard style) vs random forest
  4. Evaluate: AUC, Gini, and the business cost of wrong decisions
  5. Choose an approval cut-off that minimises that cost

Usage:  python credit_risk_analysis.py
All tables (CSV) and charts (PNG) are written next to this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
SEED = 42

# The dataset's own cost matrix: approving a bad loan costs 5,
# rejecting a good customer costs 1.
COST_BAD_APPROVED = 5
COST_GOOD_REJECTED = 1

# ---------------------------------------------------------------- 1. load
COLUMNS = ["checking_account", "duration_months", "credit_history", "purpose",
           "credit_amount", "savings", "employment_since", "installment_rate",
           "personal_status_sex", "other_debtors", "residence_since", "property",
           "age", "other_installment_plans", "housing", "existing_credits",
           "job", "dependants", "telephone", "foreign_worker", "class"]

LABELS = {
    "checking_account": {"A11": "below 0 DM", "A12": "0 to 200 DM",
                         "A13": "200 DM or more", "A14": "no checking account"},
    "credit_history": {"A30": "no credits / all paid back", "A31": "all paid back at this bank",
                       "A32": "existing credits paid so far", "A33": "delays in the past",
                       "A34": "critical / credits elsewhere"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio / TV",
                "A44": "domestic appliances", "A45": "repairs", "A46": "education",
                "A48": "retraining", "A49": "business", "A410": "other"},
    "savings": {"A61": "below 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM",
                "A64": "1000 DM or more", "A65": "unknown / none"},
    "employment_since": {"A71": "unemployed", "A72": "under 1 year", "A73": "1 to 4 years",
                         "A74": "4 to 7 years", "A75": "7 years or more"},
    "personal_status_sex": {"A91": "male, divorced/separated", "A92": "female, div./sep./married",
                            "A93": "male, single", "A94": "male, married/widowed"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings agreement / life insurance",
                 "A123": "car or other", "A124": "unknown / none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee", "A174": "management / highly qualified"},
    "telephone": {"A191": "no", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

df = pd.read_csv(HERE / "german.data", sep=" ", header=None, names=COLUMNS)
for col, mapping in LABELS.items():
    df[col] = df[col].map(mapping)
assert df.notna().all().all(), "unexpected code in the raw file"
df["default"] = (df["class"] == 2).astype(int)        # 1 = bad credit risk
df = df.drop(columns="class")

print(f"{len(df)} loans, {df['default'].mean():.1%} defaulted\n")

# ------------------------------------------------------------- 2. explore
df["duration_band"] = pd.cut(df["duration_months"], [0, 12, 24, 36, 100],
                             labels=["up to 12", "13 to 24", "25 to 36", "over 36"])
df["amount_band"] = pd.qcut(df["credit_amount"], 4,
                            labels=["lowest quarter", "second quarter",
                                    "third quarter", "highest quarter"])
df["age_band"] = pd.cut(df["age"], [0, 24, 34, 49, 100],
                        labels=["under 25", "25 to 34", "35 to 49", "50 and over"])

explore = ["checking_account", "credit_history", "savings", "duration_band",
           "amount_band", "purpose", "employment_since", "housing", "age_band"]
rows = []
for col in explore:
    g = df.groupby(col, observed=True)["default"].agg(loans="size", default_rate="mean")
    for value, r in g.iterrows():
        rows.append({"feature": col, "value": value, "loans": int(r.loans),
                     "default_rate_pct": round(100 * r.default_rate, 1)})
default_rates = pd.DataFrame(rows)
default_rates.to_csv(HERE / "default_rates.csv", index=False)
print(default_rates.to_string(index=False), "\n")

# --------------------------------------------------------------- 3. model
# Sex/marital status and foreign-worker status are left out of the model:
# a lender should not base a credit decision on them.
EXCLUDED = ["personal_status_sex", "foreign_worker"]
NUMERIC = ["duration_months", "credit_amount", "installment_rate", "residence_since",
           "age", "existing_credits", "dependants"]
CATEGORICAL = [c for c in LABELS if c not in EXCLUDED]

X, y = df[NUMERIC + CATEGORICAL], df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED)

# Most frequent category first, so it becomes the reference level
# that the logistic regression compares all other levels against.
cat_orders = [list(X_train[c].value_counts().index) for c in CATEGORICAL]


def preprocess(drop):
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(categories=cat_orders, drop=drop), CATEGORICAL)])


models = {
    "Logistic regression": Pipeline([
        ("prep", preprocess("first")),
        ("model", LogisticRegression(max_iter=1000))]),
    "Random forest": Pipeline([
        ("prep", preprocess(None)),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                         random_state=SEED))]),
}

# ------------------------------------------------------------ 4. evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
comparison, test_prob, cv_prob = [], {}, {}
for name, model in models.items():
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_prob[name] = cross_val_predict(model, X_train, y_train, cv=cv,
                                      method="predict_proba")[:, 1]
    model.fit(X_train, y_train)
    test_prob[name] = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_prob[name])
    comparison.append({"model": name,
                       "cv_auc_mean": round(cv_auc.mean(), 3),
                       "cv_auc_std": round(cv_auc.std(), 3),
                       "test_auc": round(auc, 3),
                       "test_gini": round(2 * auc - 1, 3)})
comparison = pd.DataFrame(comparison)
comparison.to_csv(HERE / "model_comparison.csv", index=False)
print(comparison.to_string(index=False), "\n")

# What drives the logistic regression: odds ratios
logit = models["Logistic regression"]
names = logit.named_steps["prep"].get_feature_names_out()
odds = pd.DataFrame({"feature": names,
                     "coefficient": logit.named_steps["model"].coef_[0]})
odds["feature"] = (odds["feature"].str.replace("num__", "").str.replace("cat__", "")
                   .str.replace("_", " "))
odds["odds_ratio"] = np.exp(odds["coefficient"]).round(2)
odds["coefficient"] = odds["coefficient"].round(3)
odds = odds.sort_values("coefficient", ascending=False)
odds.to_csv(HERE / "odds_ratios.csv", index=False)
print(pd.concat([odds.head(8), odds.tail(8)]).to_string(index=False), "\n")

# ------------------------------------------------- 5. cost-based cut-off
def decision_cost(y_true, prob, threshold):
    """Reject when predicted default probability >= threshold."""
    reject = prob >= threshold
    bad_approved = int(((y_true == 1) & ~reject).sum())
    good_rejected = int(((y_true == 0) & reject).sum())
    cost = COST_BAD_APPROVED * bad_approved + COST_GOOD_REJECTED * good_rejected
    return cost, bad_approved, good_rejected, int(reject.sum())


thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
# The cut-off is chosen on cross-validated training predictions,
# never on the test set, so the test result stays an honest estimate.
train_costs = [decision_cost(y_train.values, cv_prob["Logistic regression"], t)[0]
               for t in thresholds]
best_t = float(thresholds[int(np.argmin(train_costs))])

n_test = len(y_test)
policies = []
for label, t in [("Approve everyone", 1.01), ("Model, cut-off 0.50", 0.50),
                 (f"Model, cost-based cut-off {best_t:.2f}", best_t),
                 ("Reject everyone", 0.0)]:
    cost, ba, gr, rejected = decision_cost(y_test.values, test_prob["Logistic regression"], t)
    policies.append({"policy": label, "rejected_pct": round(100 * rejected / n_test, 1),
                     "bad_loans_approved": ba, "good_customers_rejected": gr,
                     "cost_per_100_applicants": round(100 * cost / n_test, 1)})
policies = pd.DataFrame(policies)
policies.to_csv(HERE / "decision_policies.csv", index=False)
print(policies.to_string(index=False), "\n")

pred = (test_prob["Logistic regression"] >= best_t).astype(int)
print("Confusion matrix at the cost-based cut-off (rows = actual good/bad, "
      "columns = approved/rejected):")
print(confusion_matrix(y_test, pred), "\n")

# ----------------------------------------------------------------- charts
BLUE, ORANGE, INK, SUB, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "font.size": 10, "text.color": INK, "axes.labelcolor": SUB,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlepad": 14})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=160)
    plt.close(fig)


def rate_chart(feature, title, name, order=None):
    d = default_rates[default_rates["feature"] == feature].set_index("value")
    d = d.loc[order] if order else d.sort_values("default_rate_pct")
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(d) + 1.3))
    labels = [f"{v}  (n={n})" for v, n in zip(d.index, d["loans"])]
    bars = ax.barh(labels, d["default_rate_pct"], color=BLUE, height=0.6)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in d["default_rate_pct"]], padding=4, color=SUB)
    ax.axvline(100 * y.mean(), color=MUTED, lw=1, ls="--")
    ax.text(100 * y.mean(), len(d) - 0.35, f" overall {y.mean():.0%}", color=MUTED, fontsize=9)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0, labelcolor=SUB)
    ax.set_xlim(0, d["default_rate_pct"].max() * 1.18)
    ax.invert_yaxis()
    save(fig, name)


rate_chart("checking_account", "Default rate by checking account balance",
           "default_by_checking_account.png")
rate_chart("duration_band", "Default rate by loan duration (months)",
           "default_by_duration.png", order=["up to 12", "13 to 24", "25 to 36", "over 36"])

# ROC curves
fig, ax = plt.subplots(figsize=(5.6, 5))
for (name, prob), color in zip(test_prob.items(), [BLUE, ORANGE]):
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC {roc_auc_score(y_test, prob):.2f})")
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random guess")
ax.set_title("ROC curves on the test set")
ax.set_xlabel("Share of good customers wrongly flagged")
ax.set_ylabel("Share of defaulters caught")
ax.grid(color=GRID, lw=0.8)
ax.tick_params(length=0)
ax.legend(frameon=False, loc="lower right", labelcolor=SUB)
save(fig, "roc_curves.png")

# Cost by cut-off
fig, ax = plt.subplots(figsize=(8, 4))
per100 = [100 * c / len(y_train) for c in train_costs]
ax.plot(thresholds, per100, color=BLUE, lw=2)
ax.scatter([best_t], [min(per100)], color=BLUE, s=50, zorder=3,
           edgecolor="#fcfcfb", linewidth=2)
ax.annotate(f"lowest cost at {best_t:.2f}", (best_t, min(per100)), xytext=(10, -14),
            textcoords="offset points", color=SUB)
ax.set_title("Cost per 100 applicants by rejection cut-off (cross-validated, training data)")
ax.set_xlabel("Reject when predicted default probability is at least")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.tick_params(length=0)
ax.set_ylim(0)
save(fig, "cost_by_cutoff.png")

# Strongest risk factors
top = pd.concat([odds.head(7), odds.tail(7)]).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top["feature"], top["coefficient"], height=0.6,
        color=[ORANGE if c > 0 else BLUE for c in top["coefficient"]])
ax.axvline(0, color="#c3c2b7", lw=1)
ax.set_title("Strongest risk factors in the logistic regression\n"
             "orange = higher default risk, blue = lower default risk")
ax.set_xlabel("Coefficient (log-odds, compared with the most common category)")
ax.tick_params(length=0, labelcolor=SUB)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
save(fig, "risk_factors.png")

print("Done.")
