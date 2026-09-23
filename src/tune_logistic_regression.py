import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

train_df = pd.read_csv("data/processed/train_clean.csv")
val_df = pd.read_csv("data/processed/validation.csv")

train_text = (
    train_df["sentence"]
    + " [TARGET] "
    + train_df["target"]
)

val_text = (
    val_df["sentence"]
    + " [TARGET] "
    + val_df["target"]
)

y_train = train_df["polarity"]
y_val = val_df["polarity"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_text)
X_val = vectorizer.transform(val_text)

C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
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
print(f"Validation Accuracy: {best_row['val_accuracy']:.4f}")
print(f"Validation Macro F1: {best_row['val_macro_f1']:.4f}")