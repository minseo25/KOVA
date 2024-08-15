import os
import threading
import tkinter as tk
from io import BytesIO

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
import time
from dotenv import load_dotenv
from groq import Groq
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer

import pyaudio
import wave
from faster_whisper import WhisperModel
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc


# for windows (if you have duplicate dll initialization error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# load environment variables
load_dotenv()


class VoiceAssistantApp:
    def __init__(self):
        self.models = self.initialize_models()
        self.tokenizers = self.initialize_tokenizers()
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.recording = False
        self.freeze_until = 0
        self.model_name = ''
        self.thread = None
        self.setup_audio()
        self.setup_vad()
        self.setup_hotword_detector()
        self.setup_gui()

    def initialize_models(self):
        return {
            'gemma-2-2b-it': AutoModelForCausalLM.from_pretrained(
                'models/gemma-2-2b-it',
                device_map='auto',  # allocate model to GPU if available
                torch_dtype=torch.bfloat16,
            ),
            'llama-3-ko-bllossom-int4': Llama(
                model_path='models/llama-3-korean-bllossom-8b-gguf/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf',
                n_ctx=512,
                n_gpu_layers=0,  # Number of model layers to offload to GPU
            ),
            'llama3.1-8b-instant': None,
        }

    def initialize_tokenizers(self):
        return {
            'gemma-2-2b-it': AutoTokenizer.from_pretrained('models/gemma-2-2b-it'),
            'llama-3-ko-bllossom-int4': AutoTokenizer.from_pretrained('models/llama-3-korean-bllossom-8b-gguf'),
            'llama3.1-8b-instant': None,
        }

    def setup_audio(self):
        self.frames = []
        self.FORMAT = pyaudio.paInt16  # 16-bit resolution
        self.CHANNELS = 1  # mono
        self.RATE = 44100  # 44.1kHz sampling rate
        self.CHUNK = 8192  # buffer size
        self.OUTPUT_FILENAME = 'tmp.wav'
        self.MAX_SILENT_DURATION = 2.0

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.FORMAT, channels=self.CHANNELS,
            rate=self.RATE, input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=None,
        )

    def setup_vad(self):
        self.whisper = WhisperModel('models/faster-whisper-small', device='cpu', compute_type='int8')
        segmentation = Model.from_pretrained('models/pyannote-segmentation-3/pytorch_model.bin')
        self.pipeline = VoiceActivityDetection(segmentation=segmentation)
        HYPER_PARAMETERS = {
            'min_duration_on': 0.15,
            'min_duration_off': 0.1,
        }
        self.pipeline.instantiate(HYPER_PARAMETERS)

    def setup_hotword_detector(self):
        base_model = Resnet50_Arc_loss()
        self.malbud_hw = HotwordDetector(
            hotword='himalbud',
            model=base_model,
            reference_file=os.path.join(samples_loc, 'himalbud_ref.json'),
            threshold=0.7,
            relaxation_time=2,  # hotword 감지 후, 10초동안은 추가 감지 방지
        )
        self.mic_stream = SimpleMicStream(
            window_length_secs=1.5,
            sliding_window_secs=0.75,
        )
        self.mic_stream.start_stream()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title('Enter Message')

        self.msg = tk.StringVar()
        self.msg.set('Select a model to ask')

        btn1 = tk.Button(
            self.root, text='Gemma-2-it',
            command=lambda: self.model_selected('gemma-2-2b-it'),
        )
        btn2 = tk.Button(
            self.root, text='ko-bllossom-int4',
            command=lambda: self.model_selected('llama-3-ko-bllossom-int4'),
        )
        btn3 = tk.Button(
            self.root, text='llama3.1-8b',
            command=lambda: self.model_selected('llama3.1-8b-instant'),
        )
        btn1.grid(row=0, column=0, padx=10, pady=10)
        btn2.grid(row=0, column=1, padx=10, pady=10)
        btn3.grid(row=0, column=2, padx=10, pady=10)

        self.label = tk.Label(self.root, textvariable=self.msg)
        self.label.grid(row=1, column=0, columnspan=3, pady=10)

        hotword_thread = threading.Thread(target=self.detect_hotword)
        hotword_thread.daemon = True
        hotword_thread.start()

        self.root.mainloop()

    def model_selected(self, name):
        self.msg.set(f"Ask to {name}!")
        self.model_name = name

    def record_audio(self):
        print('Recording started...')
        self.update_recording_status(True)

        self.frames.clear()
        silent_duration = 0.0

        while True:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            self.frames.append(data)

            waveform = torch.from_numpy(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
            waveform = waveform.unsqueeze(0)

            vad_result = self.pipeline({'waveform': waveform, 'sample_rate': self.RATE})

            print(f"VAD result: {vad_result.get_timeline()}")

            if vad_result.get_timeline():
                silent_duration = 0.0
            else:
                silent_duration += self.CHUNK / self.RATE

            if silent_duration >= self.MAX_SILENT_DURATION:
                break

        print('Recording completed.')
        wf = wave.open(self.OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        self.update_recording_status(False)
        self.freeze_until = time.time() + 5.0  # freeze hotword detection for 5 seconds

    def update_recording_status(self, recording):
        self.recording = recording
        if self.recording:
            self.msg.set('Recording...')
            self.label.config(fg='red', bg='white')
        else:
            self.msg.set('Ready to detect hotword.')
            self.label.config(fg='black', bg='white')

    def ask_llm(self):
        if self.model_name not in self.models:
            self.msg.set('Choose model first!')
            return

        if self.thread is not None:
            self.thread.stop()

        self.record_audio()

        segments, _ = self.whisper.transcribe(self.OUTPUT_FILENAME, beam_size=5, language='ko')
        user_input = ''.join(segment.text for segment in segments).strip()
        print(f"User input: {user_input}")

        if not user_input:
            return

        self.thread = LLM2TTSThread(user_input, self.model_name, self.models, self.tokenizers, self.device, self.client)
        self.thread.start()

    def detect_hotword(self):
        while True:
            if self.recording or time.time() < self.freeze_until:
                continue

            frame = self.mic_stream.getFrame()
            result = self.malbud_hw.scoreFrame(frame)
            if result is None:
                continue

            if result['match']:
                print('Wakeword uttered', result['confidence'])
                self.ask_llm()


class LLM2TTSThread(threading.Thread):
    def __init__(self, user_input, model_name, models, tokenizers, device, client, chunk_size=1024):
        threading.Thread.__init__(self)
        self.user_input = user_input
        self.model_name = model_name
        self.models = models
        self.tokenizers = tokenizers
        self.device = device
        self.client = client
        self.chunk_size = chunk_size
        self._stop_event = threading.Event()
        self.stream = None

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
        model = self.models[self.model_name]
        tokenizer = self.tokenizers[self.model_name]
        messages = [
            {'role': 'user', 'content': '안녕 만나서 반가워. 나는 한국어로 너에게 질문할거야. 이모지를 사용하지 말아줘.'},
            {'role': 'assistant', 'content': '안녕하세요! 저는 한국어로만 답변하는 도우미입니다. 질문을 해주세요.'},
            {'role': 'user', 'content': self.user_input},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors='pt', return_dict=True,
        ).to(self.device)
        outputs = model.generate(**input_ids, max_new_tokens=256)
        response = tokenizer.decode(
            outputs[0], skip_special_tokens=True,
        ).strip()
        return response.split(self.user_input.strip())[-1].strip() if self.user_input.strip() in response else response

    def _run_llama3_ko(self):
        model = self.models[self.model_name]
        tokenizer = self.tokenizers[self.model_name]
        PROMPT = \
            '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 한국어로 친절하고 정확하게 답변해야 합니다.
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
            'stop': [''],
            'top_p': 0.9,
            'temperature': 0.6,
            'echo': True,  # Echo the prompt in the output
        }
        response_msg = model(prompt, **generation_kwargs)
        return response_msg['choices'][0]['text'][len(prompt):].strip()

    def _run_llama3_1_8b(self):
        chat_completion = self.client.chat.completions.create(
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


if __name__ == '__main__':
    VoiceAssistantApp()
