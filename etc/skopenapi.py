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

for voice in ['aria', 'aria_call', 'aria_dj', 'jiyoung', 'juwon', 'jihun', 'hamin', 'daeho']:
    data = {
        'text': '하이 말벗',
        'voice': voice,
        'lang': 'ko-KR',
        'speed': '1.5',
        'sformat': 'wav',
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        # play audio file using sounddevice
        audio_buffer = BytesIO(response.content)
        data, samplerate = sf.read(audio_buffer)
        sd.play(data, samplerate)
        sd.wait()
        # save response content to a file
        with open(f'{voice}_fast.wav', 'wb') as f:
            f.write(response.content)
    else:
        print(response.json())
