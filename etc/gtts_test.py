from io import BytesIO

import sounddevice as sd
import soundfile as sf
from gtts import gTTS


def text_to_speech_play(text, lang='en'):
    # Convert text to speech using gtts
    tts = gTTS(text=text, lang=lang)

    # Save the converted audio to a BytesIO object
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    # Read the BytesIO object and play it using sounddevice
    data, samplerate = sf.read(audio_buffer)
    sd.play(data, samplerate)
    sd.wait()


# Example usage
text = '안녕 만나서 반가워. 나는 한국어로 너에게 질문하는 것을 좋아해. 이모지를 사용하지 말아줘.'
text_to_speech_play(text, lang='ko')
