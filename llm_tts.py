import os
import threading
import tkinter as tk
from io import BytesIO

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
from dotenv import load_dotenv
from groq import Groq
from llama_cpp import Llama
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import pyaudio
import wave
from faster_whisper import WhisperModel
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

# for windows (if you have duplicate dll initialization error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# load environment variables
load_dotenv()

# initialize tokenizers and models
tokenizers = {
    'gemma-2-2b-it': AutoTokenizer.from_pretrained('models/gemma-2-2b-it'),
    'llama-3-ko-bllossom-int4': AutoTokenizer.from_pretrained('models/llama-3-korean-bllossom-8b-gguf'),
}

models = {
    'gemma-2-2b-it': AutoModelForCausalLM.from_pretrained(
        'models/gemma-2-2b-it',
        device_map='auto',  # allocate model to GPU if available
        torch_dtype=torch.bfloat16,
    ),
    'llama-3-ko-bllossom-int4': Llama(
        model_path='models/llama-3-korean-bllossom-8b-gguf/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf',
        n_ctx=512,
        n_gpu_layers=0,        # Number of model layers to offload to GPU
    ),
}

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# recording parameters
frames: list = []
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # mono
RATE = 44100              # 44.1kHz sampline rate
CHUNK = 8192              # buffer size
OUTPUT_FILENAME = 'tmp.wav'
MAX_SILENT_DURATION = 2.0

# initialize PyAudio and open stream
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT, channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK,
    input_device_index=None,
)

# fast whisper model
whisper = WhisperModel('models/faster-whisper-small', device='cpu', compute_type='int8')

# pyannote segmentation model for VAD
segmentation = Model.from_pretrained('models/pyannote-segmentation-3/pytorch_model.bin')
pipeline = VoiceActivityDetection(segmentation=segmentation)
HYPER_PARAMETERS = {
    # remove speech regions shorter than that many seconds.
    'min_duration_on': 0.15,
    # fill non-speech regions shorter than that many seconds.
    'min_duration_off': 0.1,
}
pipeline.instantiate(HYPER_PARAMETERS)

# initialize GROQ client
# you can try gemma2, llama3.1, lamma3, mixtral, etc.
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# global variables for multithreading
thread = None
model_name = ''


# thread class for handling model queries and TTS
class LLM2TTSThread(threading.Thread):
    def __init__(self, user_input, model_name, chunk_size=1024):
        threading.Thread.__init__(self)
        self.user_input = user_input
        self.model_name = model_name
        self._stop_event = threading.Event()
        self.stream = None
        self.chunk_size = chunk_size

    def run(self):
        try:
            response = self._run_model_query()
            print('Response:', response)
            if self._stop_event.is_set():
                return
            if not response:
                response = '죄송해요, 아직 답변이 불가능한 질문이에요.'
            self._play_tts_response(response)
        except Exception as e:
            print(f"Exception in thread: {e}")

    def stop(self):
        self._stop_event.set()

    def _run_model_query(self):
        if self.model_name == 'gemma-2-2b-it':
            return self._run_gemma2()
        elif self.model_name == 'llama-3-ko-bllossom-int4':
            return self._run_llama3_ko()
        elif self.model_name == 'llama3.1-8b-instant':
            return self._run_llama3_1_8b()

    def _run_gemma2(self):
        model = models[self.model_name]
        tokenizer = tokenizers[self.model_name]
        messages = [
            {'role': 'user', 'content': '안녕 만나서 반가워. 나는 한국어로 너에게 질문할거야. 이모지를 사용하지 말아줘.'},
            {'role': 'assistant', 'content': '안녕하세요! 저는 한국어로만 답변하는 도우미입니다. 질문을 해주세요.'},
            {'role': 'user', 'content': self.user_input},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors='pt', return_dict=True,
        ).to(device)
        outputs = model.generate(**input_ids, max_new_tokens=256)
        response = tokenizer.decode(
            outputs[0], skip_special_tokens=True,
        ).strip()
        return response.split(self.user_input.strip())[-1].strip() if self.user_input.strip() in response else response

    def _run_llama3_ko(self):
        model = models[self.model_name]
        tokenizer = tokenizers[self.model_name]
        PROMPT = \
            '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
        You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
        messages = [
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': self.user_input},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        generation_kwargs = {
            'max_tokens': 512,
            'stop': ['<|eot_id|>'],
            'top_p': 0.9,
            'temperature': 0.6,
            'echo': True,  # Echo the prompt in the output
        }
        response_msg = model(prompt, **generation_kwargs)
        return response_msg['choices'][0]['text'][len(prompt):].strip()

    def _run_llama3_1_8b(self):
        chat_completion = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': 'You are a helpful korean assistant. 지금부터 너는 한국어로 대답을 할거야'},
                {'role': 'user', 'content': self.user_input},
            ],
            model='llama-3.1-8b-instant',
            max_tokens=512,
        )
        return chat_completion.choices[0].message.content.strip()

    def _play_tts_response(self, response):
        headers = {
            'Content-Type': 'application/json',
            'appKey': os.getenv('SK_OPEN_API_KEY'),
        }
        data = {
            'text': response, 'voice': 'aria',
            'lang': 'ko-KR', 'speed': '1.0', 'sformat': 'wav',
        }
        tts_response = requests.post(
            'https://apis.openapi.sk.com/tvoice/tts', headers=headers, json=data,
        )

        if self._stop_event.is_set() or tts_response.status_code != 200:
            return

        audio_buffer = BytesIO(tts_response.content)
        data, samplerate = sf.read(audio_buffer)
        data = data.astype(np.float32)
        self.stream = sd.OutputStream(
            samplerate=samplerate, channels=len(data.shape),
        )
        self.stream.start()

        for i in range(0, len(data), self.chunk_size):
            if self._stop_event.is_set():
                self.stream.abort()
                return
            self.stream.write(data[i:i+self.chunk_size])


def model_selected(msg, name):
    global model_name
    msg.set(f"Ask to {name}!")
    model_name = name


def record_audio():
    print('Recording started...')
    frames.clear()
    silent_duration = 0.0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        waveform = torch.from_numpy(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
        waveform = waveform.unsqueeze(0)

        vad_result = pipeline({'waveform': waveform, 'sample_rate': RATE})

        print(f"VAD result: {vad_result.get_timeline()}")

        if vad_result.get_timeline():
            silent_duration = 0.0
        else:
            silent_duration += CHUNK / RATE

        if silent_duration >= MAX_SILENT_DURATION:
            break

    print('Recording stopped.')

    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return


def ask_llm(msg):
    if model_name not in (
        'gemma-2-2b-it',
        'llama-3-ko-bllossom-int4',
        'llama3.1-8b-instant',
    ):
        msg.set('Choose model first!')
        return

    global thread
    if thread is not None:
        thread.stop()
        # thread.join()  # this will block the main thread

    # start recording in a separate thread
    record_audio()

    segments, info = whisper.transcribe(OUTPUT_FILENAME, beam_size=5, language='ko')
    user_input = ''
    for segment in segments:
        user_input += segment.text
    print(f"User input: {user_input}")

    if not user_input.strip():
        return

    thread = LLM2TTSThread(user_input, model_name)
    thread.start()


def main():
    root = tk.Tk()
    root.title('Enter Message')

    msg = tk.StringVar()
    msg.set('Select a model to ask')

    btn1 = tk.Button(
        root, text='Gemma-2-it',
        command=lambda: model_selected(msg, 'gemma-2-2b-it'),
    )
    btn2 = tk.Button(
        root, text='ko-bllossom-int4',
        command=lambda: model_selected(msg, 'llama-3-ko-bllossom-int4'),
    )
    btn3 = tk.Button(
        root, text='llama3.1-8b',
        command=lambda: model_selected(msg, 'llama3.1-8b-instant'),
    )
    btn1.grid(row=0, column=0, padx=10, pady=10)
    btn2.grid(row=0, column=1, padx=10, pady=10)
    btn3.grid(row=0, column=2, padx=10, pady=10)

    label = tk.Label(root, textvariable=msg)
    label.grid(row=1, column=0, columnspan=3, pady=10)
    ask_button = tk.Button(
        root, text='ask to llm',
        command=lambda: ask_llm(msg),
    )
    ask_button.grid(row=2, column=1, pady=10)

    root.mainloop()


if __name__ == '__main__':
    main()
