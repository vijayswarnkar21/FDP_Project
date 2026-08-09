from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
import numpy as np
load_dotenv()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddingModel = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
model = ChatAnthropic(model="claude-haiku-4-5")


chunks = []

embiddingsStore = []

loader = PyPDFLoader("JECRC_Overview_Embedding_Ready.pdf")
docs = loader.load()

for doc in docs:
    docChunks = text_splitter.split_text(doc.page_content)
    for chunk in docChunks:
        embiddingsStore.append({
            "chunk": chunk,
            "vector": embeddingModel.embed_query(chunk)
        })

message =  None
chats = []

SYSTEM_PROMPT_TEMPLATE = (
    "Here I am providing content which you are supposed to answer the user's query from. "
    "If the query cannot be answered based on the content, clearly print 'I do not know the answer'. "
    "Content is: {content}"
    "answer only in 5 lines not more than that"
)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def getReleventContent(message):
    messageEmbedding = embeddingModel.embed_query(message)
    consineSimilarity = [
        (cosine_similarity(messageEmbedding, embidding['vector']), embidding['chunk'])
        for embidding in embiddingsStore
    ]
    consineSimilarity.sort(key=lambda x: x[0], reverse=True)
    releventEmbedding = consineSimilarity[:10]
    releventContent = []
    for _, chunk in releventEmbedding:
        releventContent.append(chunk)
    return releventContent

while True:
    message = input("Write your query:- ")
    message = message.strip()
    releventContent = getReleventContent(message)
    systemMessage = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(content=releventContent))
    humanMessage = HumanMessage(content=message)
    chats.append(humanMessage)
    result = model.invoke([systemMessage] + chats)
    print(result.content)
    chats.append(AIMessage(content=result.content))

     






