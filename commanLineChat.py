from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chats = []

model = ChatAnthropic(model="claude-haiku-4-5")

while True:
    user = input("Enter Your Message: - ")
    message = user.strip()
    chats.append(message)
    if message.lower() == "exit":
        break
    result = model.invoke(chats)
    chats.append(result.content)
    print(result.content)    
