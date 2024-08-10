# https://github.com/coqui-ai/TTS/issues/3848
# xtts not supported in python 3.12 version
import threading

import torch
from TTS.api import TTS
# from TTS.utils.manage import ModelManager

# Get device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# List available 🐸TTS models
# print(ModelManager().list_models())

tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2').to(device)


def thread(user_input, id):
    wav = tts.tts(text=user_input, speaker_wav='womanvoice.wav', language='en')
    print(f"thread{id} wav length: {len(wav)}")


for i in range(5):
    threading.Thread(
        target=thread, args=(
            'Hi, my name is minseo kim. Who are you?', i,
        ),
    ).start()

# tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# wav = tts.tts(text="Hello world. Can you speak long sentence fluently?", speaker_wav="myvoice.wav", language="en")
# sd.play(wav, samplerate=int(22050*1.25))
# sd.wait()
