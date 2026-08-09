from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from FDP_Project.commandLineSimilarity import embedding
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddingModel = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
model = ChatAnthropic(model="claude-haiku-4-5")

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

message =  None
chats = []

SYSTEM_PROMPT_TEMPLATE = (
    "Here I am providing content which you are supposed to answer the user's query from. "
    "If the query cannot be answered based on the content, clearly print 'I do not know the answer'. "
    "Content is: {content}"
)

def getReleventContent():
    ""

while True:
    message = input("Write your query:- ")
    message = message.strip()
    releventContent = getReleventContent(message)
    systemMessage = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(content=releventContent))
    humanMessage = HumanMessage(content=message)
    chats.extend([systemMessage, humanMessage])
    result = model.invoke(chats)
    print(result.content)
    humanMessage = HumanMessage(content=result.content)
    chats.append(humanMessage)

     






