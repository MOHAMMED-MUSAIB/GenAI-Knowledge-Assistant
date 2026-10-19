
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "Python is a programming language.",
    "Python is used to develop software.",
    "The weather is sunny today.",
]

embeddings = model.encode(sentences, convert_to_tensor=True)

score_related = cos_sim(embeddings[0], embeddings[1]).item()
score_unrelated = cos_sim(embeddings[0], embeddings[2]).item()

print(f"Related similarity: {score_related:.4f}")
print(f"Unrelated similarity: {score_unrelated:.4f}")

assert score_related > score_unrelated
print("Similarity test passed!")
