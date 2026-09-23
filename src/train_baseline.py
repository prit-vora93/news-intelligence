import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


train_df = pd.read_csv("data/processed/train_clean.csv")
val_df = pd.read_csv("data/processed/validation.csv")
test_df = pd.read_csv("data/processed/test.csv")


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

test_text = (
    test_df["sentence"]
    + " [TARGET] "
    + test_df["target"]
)


y_train = train_df["polarity"]
y_val = val_df["polarity"]
y_test = test_df["polarity"]


print("Training examples:", len(train_text))
print("Validation examples:", len(val_text))
print("Test examples:", len(test_text))


vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train = vectorizer.fit_transform(train_text)
X_val = vectorizer.transform(val_text)
X_test = vectorizer.transform(test_text)


print("\nTF-IDF Shapes")
print("-------------")
print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")


model = LogisticRegression(C = 1, max_iter=1000,)

model.fit(X_train, y_train)


val_predictions = model.predict(X_val)

val_accuracy = accuracy_score(y_val, val_predictions)

cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=[2.0, 4.0, 6.0],
)

print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

# print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

print("\nConfusion Matrix")
print("----------------")
print(cm)

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_val,
        val_predictions,
        labels=[2.0, 4.0, 6.0],
        target_names=["Negative", "Neutral", "Positive"],
    )
)