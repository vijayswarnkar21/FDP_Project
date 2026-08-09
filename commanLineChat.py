from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

chats = []

model = ChatAnthropic(model="claude-haiku-4-5")

while True:
    user = input("Enter Your Message: - ")
    message = user.strip()
    chats.append(HumanMessage(content=message))
    if message.lower() == "exit":
        break
    result = model.invoke(chats)
    chats.append(AIMessage(content=result.content))
    print("--------------------------")
    print(chats)
    print("--------------------------")
    print(result.content)    
