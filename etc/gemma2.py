import threading
import warnings

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import logging

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained('../models/gemma-2-2b-it')
model = AutoModelForCausalLM.from_pretrained(
    '../models/gemma-2-2b-it',
    device_map='auto',  # allocate model to GPU if available
    torch_dtype=torch.bfloat16,
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def thread(user_input, id):
    messages = [
        {'role': 'user', 'content': '안녕 만나서 반가워. 나는 한국어로 너에게 질문할거야. 이모지를 사용하지 말아줘.'},
        {'role': 'assistant', 'content': '안녕하세요! 저는 한국어로만 답변하는 도우미입니다. 질문을 해주세요.'},
        {'role': 'user', 'content': user_input},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors='pt', return_dict=True,
    ).to(device)
    outputs = model.generate(**input_ids, max_new_tokens=256)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if user_input.strip() in response:
        response = response.split(user_input.strip())[-1].strip()
    else:
        response = response.strip()

    if response == '':
        response = '죄송해요. 아직 대답할 수 없는 질문이에요.'

    print(f"thread{id}: {response}")


# test multi-threading
for i in range(5):
    threading.Thread(target=thread, args=('서울대학교에 대해 알고 있니?', i)).start()
