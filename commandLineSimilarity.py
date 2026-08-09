from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddingModel = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

embiddingsStore = []

aboutCricketers = [
    "Virat Kohli is a cricketer who plays for the India national team. He is a right-handed batsman and a right-arm medium-fast bowler.",
    "Rohit Sharma is a cricketer who plays for the India national team. He is a right-handed batsman and a right-arm medium-fast bowler.",
    "MS Dhoni is a cricketer who plays for the India national team. He is a right-handed batsman and a right-arm medium-fast bowler.",
    "Lionel Messi is a footballer who plays for the Argentina national team. He is known for his dribbling and left-footed finishing.",
    "The Eiffel Tower is a famous landmark located in Paris, France, built in 1889.",
]

for cricketer in aboutCricketers:
    embedding = embeddingModel.embed_query(cricketer)
    embiddingsStore.append(embedding)

query = input("Enter a query: ")

queryEmbedding = embeddingModel.embed_query(query)

similarityScores = []

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i, embidding in enumerate(embiddingsStore):
    similarityScores.append((cosine_similarity(queryEmbedding, embidding), i))

similarityScores.sort(key=lambda x: x[0], reverse=True)

for score, i in similarityScores:
    print(f"{score:.4f} - {aboutCricketers[i]}")
