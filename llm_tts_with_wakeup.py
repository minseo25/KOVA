import os
import threading
import time
import base64
import io
import pyaudio
import wave
import torch
import tkinter as tk
import numpy as np
import sounddevice as sd
import soundfile as sf

from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageGrab
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc

# LLM2TTSThread class
from LLM2TTSThread import LLM2TTSThread

# for windows (if you have duplicate dll initialization error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# load environment variables
load_dotenv()

# STT : Groq API
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# record parameters
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # mono
RATE = 44100              # 44.1kHz sampline rate
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

# hotword detection using eff_word_net
base_model = Resnet50_Arc_loss()
malbud_hw = HotwordDetector(
    hotword='himalbud',
    model=base_model,
    reference_file=os.path.join(samples_loc, 'himalbud_ref.json'),
    threshold=0.7,
    relaxation_time=2,  # hotword 감지 후, 10초동안은 추가 감지 방지
)
mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75,
)
mic_stream.start_stream()

# global variables
thread = None
recording = False
freeze_until = 0.0
msg = None
label = None
use_clipboard = None


def record_audio():
    global freeze_until

    # set recording status and update the GUI label
    update_recording_status(True)
    frames = []
    silent_duration = 0.0
    print('Recording started...')

    try:
        while True:
            # read audio data from the stream
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            # convert audio data to torch tensor
            waveform = torch.from_numpy(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
            waveform = waveform.unsqueeze(0)

            vad_result = pipeline({'waveform': waveform, 'sample_rate': RATE})
            print(f"VAD result: {vad_result.get_timeline()}")

            # reset silent duration if speech detected
            if vad_result.get_timeline():
                silent_duration = 0.0
            else:
                silent_duration += CHUNK / RATE
            # stop recording if silent duration exceeds the threshold
            if silent_duration >= MAX_SILENT_DURATION:
                break
    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        print('Recording stopped.')
        # stop the stream and save the audio file
        wf = wave.open(OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # update GUI label & freeze hotword detection for 5 seconds
        update_recording_status(False)
        freeze_until = time.time() + 5.0


def update_recording_status(status):
    global recording
    recording = status

    if recording:
        msg.set('Recording...')
        label.config(fg='red')
    else:
        msg.set('Ready to detect hotword.')
        label.config(fg='black')


def stt_process():
    try:
        with open(OUTPUT_FILENAME, 'rb') as audio_file:
            audio_bytes = audio_file.read()

            # use groq API for STT
            transcription = groq_client.audio.transcriptions.create(
                file=(OUTPUT_FILENAME, audio_bytes),
                model='whisper-large-v3',
                language='ko',
                temperature=0.0,
            )
            user_input = transcription.text.strip()

        # remove temporary audio file
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


def ask_llm():
    global thread, use_clipboard
    if thread is not None:
        thread.stop()
        # thread.join()  # this will block the main thread

    # start recording in a separate thread
    record_audio()
    # STT
    user_input = stt_process()

    if len(user_input) == 0:
        return

    if use_clipboard.get():
        mime_type, content = get_clipboard_content()
        thread = LLM2TTSThread(user_input, mime_type, content)
    else:
        thread = LLM2TTSThread(user_input, 'text/plain', '')
    thread.start()


def detect_hotword():
    global recording, freeze_until
    while True:
        if recording or time.time() < freeze_until:
            continue

        frame = mic_stream.getFrame()
        result = malbud_hw.scoreFrame(frame)
        if result is not None and result['match']:
            print('Hotword detected!', result['confidence'])
            # call greetings for the very first hotword detection
            if thread is None:
                filename = 'assets/greetings.wav'
                data, samplerate = sf.read(filename)

                sd.play(data, samplerate)
                sd.wait()

            ask_llm()


def main():
    global msg, label, use_clipboard

    root = tk.Tk()
    root.title('Ask to MAL-BUD!')
    img = tk.PhotoImage(file='./assets/robot.png')
    img_label = tk.Label(root, image=img)
    img_label.grid(row=0, column=0, columnspan=3, pady=10, padx=10)

    # Label for the checkbox
    clipboard_label = tk.Label(root, text='클립보드 내용도 전송')
    clipboard_label.grid(row=2, column=1)

    # Checkbox for clipboard content
    use_clipboard = tk.BooleanVar()  # Holds the state of the checkbox (True/False)
    clipboard_checkbox = tk.Checkbutton(root, variable=use_clipboard)
    clipboard_checkbox.grid(row=3, column=1)

    msg = tk.StringVar()
    msg.set('Ready to detect hotword.')
    label = tk.Label(root, textvariable=msg)
    label.grid(row=4, column=0, columnspan=3, pady=10)

    # hotword detection thread
    hotword_thread = threading.Thread(target=detect_hotword)
    hotword_thread.daemon = True
    hotword_thread.start()

    root.mainloop()


if __name__ == '__main__':
    main()
