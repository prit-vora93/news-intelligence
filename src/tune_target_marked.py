import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

train_df = pd.read_csv(
    "data/processed/train_target.csv"
)

val_df = pd.read_csv(
    "data/processed/validation_target.csv"
)

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

val_text = val_df.apply(
    mark_target,
    axis=1,
)

y_train = train_df["polarity"]
y_val = val_df["polarity"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_text)
X_val = vectorizer.transform(val_text)

print("\nTF-IDF Shapes")
print("-------------")
print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")

C_values = [
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
]

results = []

for C in C_values:

    print(f"\nTesting C = {C}")


    model = LogisticRegression(
        C=C,
        max_iter=1000,
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions,
    )

    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
    )

    results.append(
        {
            "C": C,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
        }
    )

results_df = pd.DataFrame(results)

print("\nHyperparameter Results")
print("---------------------")

print(
    results_df.to_string(
        index=False,
        formatters={
            "train_accuracy": "{:.4f}".format,
            "val_accuracy": "{:.4f}".format,
            "val_macro_f1": "{:.4f}".format,
        },
    )
)

best_row = results_df.loc[
    results_df["val_macro_f1"].idxmax()
]

print("\nBest Configuration")
print("-------------------")

print(f"C: {best_row['C']}")
print(
    f"Validation Accuracy: "
    f"{best_row['val_accuracy']:.4f}"
)
print(
    f"Validation Macro F1: "
    f"{best_row['val_macro_f1']:.4f}"
)
best_C = best_row["C"]

model = LogisticRegression(
    C=best_C,
    max_iter=1000,
)

model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
# train accuracy and validation accuracy for best model
train_accuracy = accuracy_score(
    y_train,
    train_predictions,
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions,
)

print("\nModel Evaluation")
print("----------------")

print(f"Training Accuracy:   {train_accuracy:.4f}")
print(f"Training Accuracy:   {train_accuracy * 100:.2f}%")

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")


#confusion matrix for best model
cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=[2.0, 4.0, 6.0],
)

print("\nConfusion Matrix")
print("----------------")
print(cm)

#classification report for best model

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_val,
        val_predictions,
        labels=[2.0, 4.0, 6.0],
        target_names=[
            "Negative",
            "Neutral",
            "Positive",
        ],
    )
)

