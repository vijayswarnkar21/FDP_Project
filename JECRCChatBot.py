from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

loader = PyPDFLoader("JECRC_Overview_Embedding_Ready.pdf")
docs = loader.load()
print(docs[0])
