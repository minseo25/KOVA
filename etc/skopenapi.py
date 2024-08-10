import os
from io import BytesIO

import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

url = 'https://apis.openapi.sk.com/tvoice/tts'
api_key = os.getenv('SK_OPEN_API_KEY')

headers = {
    'Content-Type': 'application/json',
    'appKey': api_key,
}
data = {
    'text': '안녕하세요, 음성합성 테스트입니다. attention X 화이팅~',
    'voice': 'aria',
    'lang': 'ko-KR',
    'speed': '1.0',
    'sformat': 'wav',
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    # play audio file using sounddevice
    audio_buffer = BytesIO(response.content)
    data, samplerate = sf.read(audio_buffer)
    sd.play(data, samplerate)
    sd.wait()
else:
    print(response.json())
