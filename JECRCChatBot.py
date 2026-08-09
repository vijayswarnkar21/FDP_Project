from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from FDP_Project.commandLineSimilarity import embedding


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddingModel = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

load_dotenv()
chunks = []

embiddingsStore = []

loader = PyPDFLoader("JECRC_Overview_Embedding_Ready.pdf")
docs = loader.load()

for doc in docs:
    chunk = text_splitter.split_text(doc.page_content)
    embeddings = {
        "chunk": chunk,
        "vector": embeddingModel.embed_query(chunk)
    }
    embiddingsStore.append(embeddings)
     






