import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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

print("\nTarget Marking Examples")
print("-----------------------")

for text in train_text.head(5):
    print(text)

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

model = LogisticRegression(
    C=2.0,
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

print("\nResults")
print("-------")
print(f"Training Accuracy:    {train_accuracy:.4f}")
print(f"Validation Accuracy:  {val_accuracy:.4f}")
print(f"Validation Macro F1:  {val_macro_f1:.4f}")