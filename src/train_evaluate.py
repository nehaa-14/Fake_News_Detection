"""
Fake News Detection on Twitter — training + evaluation pipeline.
Dataset: CONSTRAINT@AAAI2021 COVID-19 Fake News Detection (English)
Source: https://github.com/diptamath/covid_fake_news
"""
import re
import json
import string
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

RANDOM_STATE = 42
DATA_DIR = "/home/claude/fake_news_project/data"
OUT_DIR = "/home/claude/fake_news_project/outputs"

import os
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

# ---------- 1. Load ----------
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)
print(train_df['label'].value_counts())

# ---------- 2. Preprocessing ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")

def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)          # keep hashtag word, drop symbol
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

for df in (train_df, val_df, test_df):
    df["clean_tweet"] = df["tweet"].apply(clean_tweet)
    df["label_bin"] = (df["label"].str.lower() == "fake").astype(int)  # fake=1, real=0

# Train on train+val combined (as the val set is small and pre-split for a leaderboard
# task we don't need); evaluate strictly on held-out test.csv
full_train_df = pd.concat([train_df, val_df], ignore_index=True)

X_train_text = full_train_df["clean_tweet"]
y_train = full_train_df["label_bin"].values
X_test_text = test_df["clean_tweet"]
y_test = test_df["label_bin"].values

# ---------- 3. Feature engineering ----------
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    stop_words="english",
)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)
print("TF-IDF vocab size:", len(tfidf.vocabulary_))

# ---------- 4. Models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=5, random_state=RANDOM_STATE),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1, random_state=RANDOM_STATE), cv=3),
    "Naive Bayes": MultinomialNB(alpha=0.3),
}

results = {}
roc_data = {}
preds_store = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc
    }
    roc_data[name] = roc_curve(y_test, y_proba)
    preds_store[name] = (y_pred, y_proba)

    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=["real", "fake"]))

# ---------- 5. Save results table ----------
results_df = pd.DataFrame(results).T
results_df = results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4)
results_df.to_csv(f"{OUT_DIR}/results_table.csv")
print("\n=== RESULTS TABLE ===")
print(results_df)

best_model_name = results_df["f1"].idxmax()
print("\nBest model by F1:", best_model_name)

# ---------- 6. Confusion matrices ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (name, (y_pred, _)) in zip(axes, preds_store.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["real", "fake"], yticklabels=["real", "fake"], ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/confusion_matrices.png", dpi=160)
plt.close()

# ---------- 7. ROC curves ----------
plt.figure(figsize=(6, 5.5))
for name, (fpr, tpr, _) in roc_data.items():
    auc_val = results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/roc_curves.png", dpi=160)
plt.close()

# ---------- 8. Sample predictions ----------
best_pred, best_proba = preds_store[best_model_name]
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(test_df), 8, replace=False)
sample_rows = []
label_map = {0: "real", 1: "fake"}
for i in sample_idx:
    sample_rows.append({
        "tweet": test_df["tweet"].iloc[i][:140],
        "true_label": label_map[y_test[i]],
        "predicted_label": label_map[best_pred[i]],
        "confidence": round(float(best_proba[i] if best_pred[i] == 1 else 1 - best_proba[i]), 3),
        "correct": bool(y_test[i] == best_pred[i])
    })
sample_df = pd.DataFrame(sample_rows)
sample_df.to_csv(f"{OUT_DIR}/sample_predictions.csv", index=False)
print("\n=== SAMPLE PREDICTIONS ===")
print(sample_df)

# ---------- 9. Save metadata for README ----------
meta = {
    "train_size": int(len(full_train_df)),
    "test_size": int(len(test_df)),
    "class_balance_train": full_train_df["label"].value_counts().to_dict(),
    "class_balance_test": test_df["label"].value_counts().to_dict(),
    "vocab_size": len(tfidf.vocabulary_),
    "best_model": best_model_name,
    "results": results,
}
with open(f"{OUT_DIR}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nDone. Outputs saved to", OUT_DIR)
