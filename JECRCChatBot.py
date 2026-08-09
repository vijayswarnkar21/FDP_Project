from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

load_dotenv()

chunks = []

loader = PyPDFLoader("JECRC_Overview_Embedding_Ready.pdf")
docs = loader.load()

for doc in docs:
    chunk = text_splitter.split_text(doc.page_content)
    chunks.append(chunk)  





