import os
import threading
import requests
import webbrowser
import copy
import torch
import warnings
import base64
import io
import sounddevice as sd
import soundfile as sf
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from groq import Groq
from io import BytesIO
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# load environment variables
load_dotenv()

# STT and LLM(VLM) : Groq API
# TODO: use local LLM and VLM models for inference
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
# TTS : SK Open API
# TODO: use local TTS model for inference
SK_APP_KEY = os.getenv('SK_OPEN_API_KEY')

global_tokenizer = None
global_model = None
global_image_processor = None
global_max_length = None
global_device = None


def init_kollava_onevision():
    global global_tokenizer, global_model, global_image_processor, global_max_length, global_device

    warnings.filterwarnings('ignore')
    pretrained = os.path.join(
        'ckpt', 'kollava_onevision-google_siglip-so400m-patch14-384-'
        'Qwen_Qwen2.5-1.5B-Instruct-mlp2x_gelu-finetune-1.5v', 'checkpoint-1900',
    )
    # pretrained = os.path.join(
    #     'ckpt', 'kollava_onevision-google_siglip-so400m-patch14-384-'
    #     'Qwen_Qwen2.5-3B-Instruct-mlp2x_gelu-finetune-final', 'checkpoint-150',
    # )
    model_name = 'llava_qwen'

    global_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device_map = global_device
    print(f"Using device: {global_device} with device map: {device_map}")
    global_tokenizer, global_model, global_image_processor, global_max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, attn_implementation='sdpa',
    )

    global_model.eval()


# initialize kollava_onevision model only once when the script starts
init_kollava_onevision()

# thread class for handling model inference and TTS


class LLM2TTSThread(threading.Thread):
    def __init__(
        self, user_input, mime_type, content, chunk_size=1024,
        start_animation_callback=None, stop_animation_callback=None,
    ):
        threading.Thread.__init__(self)
        self.user_input = user_input
        self.mime_type = mime_type
        self.content = content
        self._stop_event = threading.Event()
        self.stream = None
        self.chunk_size = chunk_size
        self.start_animation_callback = start_animation_callback
        self.stop_animation_callback = stop_animation_callback
        self.device = global_device
        self.tokenizer = global_tokenizer
        self.model = global_model
        self.image_processor = global_image_processor
        self.max_length = global_max_length

    def run(self):
        response = self._run_model_query()

        if not self._stop_event.is_set():
            if not response:
                response = '죄송해요, 아직 답변이 불가능한 질문이에요.'

            # if response is html
            if self._is_html_response(response):
                self._open_html(response)
                response = '에이치티엠엘 파일을 띄워드리겠습니다'
            else:
                # TODO: fine-tune the response to make it more natural (maybe using a light-weight language model)
                pass
            print('Response:', response)
            self._tts_process(response)

        self.stop_animation_callback()

    def _open_html(self, response):
        with open('response.html', 'w') as f:
            f.write(response)

        # open the file in the default web browser
        webbrowser.open(f'file://{os.path.abspath("response.html")}')

    def stop(self):
        self._stop_event.set()

    def _run_model_query(self):
        if self.mime_type == 'image/jpeg':
            print('Ask to VLM')
            return self._run_llava_onevision(self.content)
        else:
            print('Ask to LLM')
            return self._run_qwen(self.content)

    def _run_llama3_vision(self, content):
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
                                'text': self.user_input,
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
                temperature=0.8,
                max_tokens=150,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                stream=False,
                stop=None,
            )
            llm_response = completion.choices[0].message.content.strip()
            return llm_response
        except Exception as e:
            print(f"LLM request error: {e}")
            return ''

    def _get_device_map(self):
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    def _is_html_response(self, response):
        soup = BeautifulSoup(response, 'html.parser')
        tags = soup.find_all(True)

        return len(tags) > 0

    def _run_llava_onevision(self, content):
        image_data = base64.b64decode(content)
        image = Image.open(io.BytesIO(image_data))
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        conv_template = 'qwen_1_5'  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + f'\n{self.user_input}'
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors='pt',
        ).unsqueeze(0).to(self.device)
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=150,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        response = text_outputs[0]
        if not self._is_html_response(response):
            response = '.'.join(response.split('.')[:-1]) + '.'
        return response

    def _run_qwen(self, content):
        # TODO: use local qwen2.5-1.5B model for inference
        prompt = ''
        if len(content) > 0:
            prompt += f"다음은 참고해야 할 정보입니다:\n{content}\n\n"
        prompt += f"질문: {self.user_input}\n답변:"

        try:
            completion = groq_client.chat.completions.create(
                messages=[
                    {'role': 'system', 'content': '당신은 유용한 한국어 비서입니다. 사용자에게 친절하고 정확하게 대답하세요.'},
                    {'role': 'user', 'content': prompt},
                ],
                model='llama-3.1-8b-instant',
                max_tokens=150,
                temperature=0.8,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5,
            )
            llm_response = completion.choices[0].message.content.strip()
            llm_response = '.'.join(llm_response.split('.')[:-1]) + '.'

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
            self.start_animation_callback()
            audio_buffer = BytesIO(tts_response.content)
            data, samplerate = sf.read(audio_buffer, dtype='float32')

            with sd.OutputStream(samplerate=samplerate, channels=len(data.shape)) as stream:
                # write audio data in chunks
                # keep checking if the stop event is set
                for i in range(0, len(data), self.chunk_size):
                    if self._stop_event.is_set():
                        stream.abort()
                        break
                    stream.write(data[i:i+self.chunk_size])

        except Exception as e:
            print(f"TTS process error: {e}")
            return
