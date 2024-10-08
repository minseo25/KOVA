import os
import torch
import wave
import pyaudio
import tkinter as tk
import base64
import io
import numpy as np

from PIL import Image, ImageGrab
from dotenv import load_dotenv
from groq import Groq
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


# load environment variables
load_dotenv()

# STT : Groq API
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# record parameters
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # mono
RATE = 44100              # 44.1kHz sampling rate
CHUNK = 8192              # buffer size
OUTPUT_FILENAME = 'tmp.wav'
MAX_SILENT_DURATION = 2.0

# initialize pyaudio object and stream
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT, channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK,
    input_device_index=None,
)

# pyannote segmentation model for VAD
# use segmentation instead of segmentation-3 to set onset and offset hyperparameters
segmentation = Model.from_pretrained('models/pyannote-segmentation/pytorch_model.bin')
pipeline = VoiceActivityDetection(segmentation=segmentation)
HYPER_PARAMETERS = {
    # mark regions as active(=speech) when probability is higher than onset value
    'onset': 0.92,
    # mark regions as inactive(=non-speech) when probability is lower than offset value
    'offset': 0.92,
    # remove speech regions shorter than that many seconds.
    'min_duration_on': 0.15,
    # fill non-speech regions shorter than that many seconds.
    'min_duration_off': 0.05,
}
pipeline.instantiate(HYPER_PARAMETERS)


def record_audio():
    frames = []
    silent_duration = 0.0
    print('Recording started...')

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            waveform = torch.from_numpy(
                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0,
            ).unsqueeze(0)

            vad_result = pipeline({'waveform': waveform, 'sample_rate': RATE})
            print(f"VAD result: {vad_result.get_timeline()}")

            if vad_result.get_timeline():
                silent_duration = 0.0
            else:
                silent_duration += CHUNK / RATE
            if silent_duration >= MAX_SILENT_DURATION:
                break
    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        print('Recording stopped.')
        wf = wave.open(OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()


def stt_process():
    try:
        with open(OUTPUT_FILENAME, 'rb') as audio_file:
            audio_bytes = audio_file.read()

            # STT 처리
            transcription = groq_client.audio.transcriptions.create(
                file=(OUTPUT_FILENAME, audio_bytes),
                model='whisper-large-v3',
                language='ko',
                temperature=0.0,
            )
            user_input = transcription.text.strip()

        os.remove(OUTPUT_FILENAME)

        return user_input
    except Exception as e:
        print(f"STT process error: {e}")
        return ''


def get_clipboard_content():
    image = ImageGrab.grabclipboard()
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        img_bytes = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return 'image/jpeg', img_bytes
    else:
        root = tk.Tk()
        root.withdraw()
        try:
            clipboard_content = root.clipboard_get()
        except Exception:
            clipboard_content = ''
        finally:
            root.destroy()

        return 'text/plain', clipboard_content
