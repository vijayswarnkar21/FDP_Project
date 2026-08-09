from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5")

while True:
    user = input("Enter Your Message: - ")
    message = user.strip()
    if message.lower() == "exit":
        break
    result = model.invoke(message)
    print(result.content)    
