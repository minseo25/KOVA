import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful korean assistant. 지금부터 너는 한국어로 대답을 할거야',
        },
        {
            'role': 'user',
            'content': '빠른 언어모델의 중요성에 대해 설명해줘',
        },
    ],
    model='llama-3.1-8b-instant',
    max_tokens=512,
)

print(chat_completion.choices[0].message.content)
