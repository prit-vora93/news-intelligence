import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


train_df = pd.read_csv(
    "data/processed/train_clean.csv"
)

val_df = pd.read_csv(
    "data/processed/validation.csv"
)


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


model = LogisticRegression(
    C=2.0,
    max_iter=1000,
)

model.fit(X_train, y_train)


val_predictions = model.predict(X_val)


errors = val_df.copy()

errors["predicted"] = val_predictions


errors = errors[
    errors["polarity"] != errors["predicted"]
]


def label_name(label):
    if label == 2.0:
        return "Negative"
    if label == 4.0:
        return "Neutral"
    if label == 6.0:
        return "Positive"
    return "Unknown"


errors["actual_label"] = errors["polarity"].apply(
    label_name
)

errors["predicted_label"] = errors["predicted"].apply(
    label_name
)


print("\nError Analysis")
print("--------------")
print(f"Validation examples: {len(val_df)}")
print(f"Incorrect predictions: {len(errors)}")


print("\nSample Errors")
print("-------------")

print(
    errors[
        [
            "sentence",
            "target",
            "actual_label",
            "predicted_label",
        ]
    ].head(20).to_string(index=False)
)


print("\nError Patterns")
print("--------------")

print(
    errors.groupby(
        ["actual_label", "predicted_label"]
    ).size()
)