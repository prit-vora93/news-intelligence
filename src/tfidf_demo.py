from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


sentences = [
    "Apple reported strong profits.",
    "Apple reported weak profits.",
    "The weather is beautiful today."
]


vectorizer = TfidfVectorizer()

matrix = vectorizer.fit_transform(sentences)


print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix Shape:")
print(matrix.shape)

print("\nTF-IDF Matrix:")
print(matrix.toarray())

# cosine similarity xheking
similarity = cosine_similarity(matrix)

print("\nCosine Similarity:")
print(similarity)