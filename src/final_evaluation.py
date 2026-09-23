import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

train_df = pd.read_csv(
    "data/processed/train_target.csv"
)

test_df = pd.read_csv(
    "data/processed/test_target.csv"
)


print("Dataset Shapes")
print("--------------")
print(f"Train: {train_df.shape}")
print(f"Test:  {test_df.shape}")


# --------------------------------------------------
# 2. Target marking
# --------------------------------------------------

def mark_target(row):
    sentence = row["sentence"]
    start = int(row["target_from"])
    end = int(row["target_to"])

    return (
        sentence[:start]
        + " [TARGET] "
        + sentence[start:end]
        + " [/TARGET] "
        + sentence[end:]
    )


train_text = train_df.apply(
    mark_target,
    axis=1,
)

test_text = test_df.apply(
    mark_target,
    axis=1,
)


y_train = train_df["polarity"]
y_test = test_df["polarity"]


# --------------------------------------------------
# 3. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(
    train_text
)

X_test = vectorizer.transform(
    test_text
)


print("\nTF-IDF Shapes")
print("-------------")
print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")


# --------------------------------------------------
# 4. Train final model
# --------------------------------------------------

model = LogisticRegression(
    C=0.5,
    max_iter=1000,
)

model.fit(
    X_train,
    y_train,
)


# --------------------------------------------------
# 5. Predictions
# --------------------------------------------------

train_predictions = model.predict(
    X_train
)

test_predictions = model.predict(
    X_test
)


# --------------------------------------------------
# 6. Accuracy
# --------------------------------------------------

train_accuracy = accuracy_score(
    y_train,
    train_predictions,
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)


test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
)


print("\nFinal Model Evaluation")
print("----------------------")

print(
    f"Training Accuracy: "
    f"{train_accuracy:.4f}"
)

print(
    f"Training Accuracy: "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{test_macro_f1:.4f}"
)


# --------------------------------------------------
# 7. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[2.0, 4.0, 6.0],
)


print("\nConfusion Matrix")
print("----------------")
print(cm)


# --------------------------------------------------
# 8. Classification Report
# --------------------------------------------------

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[2.0, 4.0, 6.0],
        target_names=[
            "Negative",
            "Neutral",
            "Positive",
        ],
    )
)