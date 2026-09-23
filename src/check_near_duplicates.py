import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(
    "NewsMTSC/NewsSentiment/experiments/default/"
    "datasets/newsmtsc-rw"
)

TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"


def load_sentences(path):
    sentences = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            sentences.append(record["sentence_normalized"])

    return sentences


train_sentences = load_sentences(TRAIN_PATH)
test_sentences = load_sentences(TEST_PATH)


print(f"Training sentences: {len(train_sentences)}")
print(f"Test sentences:     {len(test_sentences)}")


vectorizer = TfidfVectorizer()

train_matrix = vectorizer.fit_transform(train_sentences)
test_matrix = vectorizer.transform(test_sentences)



print(f"Training TF-IDF shape: {train_matrix.shape}")
print(f"Test TF-IDF shape:     {test_matrix.shape}")

similarity_matrix = cosine_similarity(test_matrix, train_matrix)
print(f"Similarity matrix shape: {similarity_matrix.shape}")


best_train_indices = similarity_matrix.argmax(axis=1)
best_similarities = similarity_matrix.max(axis=1)

top_indices = np.argsort(best_similarities)[::-1]

print("\nTop 20 Most Similar Test-Train Pairs")
print("------------------------------------")

for rank, test_index in enumerate(top_indices[:20], start=1):
    train_index = best_train_indices[test_index]
    similarity = best_similarities[test_index]

    print(f"\nRank {rank}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Test:  {test_sentences[test_index]}")
    print(f"Train: {train_sentences[train_index]}")


# print("\nTop Similar Training Sentence")
# print("-----------------------------")

# for test_index in range(10):
#     train_index = best_train_indices[test_index]
#     similarity = best_similarities[test_index]

#     print(f"\nTest sentence {test_index}")
#     print(f"Similarity: {similarity:.4f}")
#     print(f"Test:  {test_sentences[test_index]}")
#     print(f"Train: {train_sentences[train_index]}")