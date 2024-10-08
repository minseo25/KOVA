import os
import threading
import requests
import sounddevice as sd
import soundfile as sf

from dotenv import load_dotenv
from groq import Groq
from io import BytesIO

# load environment variables
load_dotenv()

# STT and LLM(VLM) : Groq API
# TODO: use local LLM and VLM models for inference
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
# TTS : SK Open API
# TODO: use local TTS model for inference
SK_APP_KEY = os.getenv('SK_OPEN_API_KEY')

# thread class for handling model inference and TTS


class LLM2TTSThread(threading.Thread):
    def __init__(self, user_input, mime_type, content, chunk_size=1024):
        threading.Thread.__init__(self)
        self.user_input = user_input
        self.mime_type = mime_type
        self.content = content
        self._stop_event = threading.Event()
        self.stream = None
        self.chunk_size = chunk_size

    def run(self):
        response = self._run_model_query()
        print('Response:', response)
        if self._stop_event.is_set():
            return
        if not response:
            response = '죄송해요, 아직 답변이 불가능한 질문이에요.'
        self._tts_process(response)

    def stop(self):
        self._stop_event.set()

    def _run_model_query(self):
        if self.mime_type == 'image/jpeg':
            print('Ask to VLM')
            return self._run_llava_onevision(self.content)
        else:
            print('Ask to LLM')
            return self._run_qwen(self.content)

    def _run_llava_onevision(self, content):
        # TODO: use local llava-onevision model for inference
        image_data_url = f"data:image/jpeg;base64,{content}"
        try:
            completion = groq_client.chat.completions.create(
                model='llama-3.2-11b-vision-preview',
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': '이 사진엔 무엇이 있니? Answer in Korean in 150 characters or less. 한국어로 150자 이내로 답변해줘.',
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': image_data_url,
                                },
                            },
                        ],
                    },
                ],
                temperature=1.0,
                max_tokens=150,
                top_p=1.0,
                stream=False,
                stop=None,
            )
            llm_response = completion.choices[0].message.content.strip()
            return llm_response
        except Exception as e:
            print(f"LLM request error: {e}")
            return ''

    def _run_qwen(self, content):
        # TODO: use local qwen2.5-1.5B model for inference
        prompt = ''
        if len(content) > 0:
            prompt += f"앞으로의 질문에 대한 답을 할 때 이 내용을 참고해줘: {content} \n\n자 그러면, "
        prompt += self.user_input

        try:
            completion = groq_client.chat.completions.create(
                messages=[
                    {'role': 'system', 'content': 'You are a helpful korean assistant. 지금부터 너는 한국어로 대답을 할거야'},
                    {'role': 'user', 'content': prompt},
                ],
                model='llama-3.1-8b-instant',
                max_tokens=150,
            )
            llm_response = completion.choices[0].message.content.strip()
            return llm_response
        except Exception as e:
            print(f"LLM request error: {e}")
            return ''

    def _tts_process(self, response):
        speakers = ['aria', 'aria_dj', 'jiyoung', 'juwon', 'jihun', 'hamin']
        try:
            tts_response = requests.post(
                'https://apis.openapi.sk.com/tvoice/tts',
                headers={
                    'Content-Type': 'application/json',
                    'appKey': SK_APP_KEY,
                },
                json={
                    'text': response, 'voice': speakers[2],
                    'lang': 'ko-KR', 'speed': '1.0', 'sformat': 'wav',
                },
            )
            if self._stop_event.is_set() or tts_response.status_code != 200:
                print('TTS request failed.')
                return

            # play audio
            audio_buffer = BytesIO(tts_response.content)
            data, samplerate = sf.read(audio_buffer, dtype='float32')

            with sd.OutputStream(samplerate=samplerate, channels=len(data.shape)) as stream:
                stream.write(data)
                if self._stop_event.is_set():
                    stream.abort()

        except Exception as e:
            print(f"TTS process error: {e}")
            return
