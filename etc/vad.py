import tkinter as tk

import numpy as np
import threading
import pyaudio
import torch
import wave
from faster_whisper import WhisperModel
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


# recording parameters
frames: list = []
is_recording = True
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # mono
RATE = 44100              # 44.1kHz sampline rate
CHUNK = 8192              # buffer size
OUTPUT_FILENAME = 'tmp.wav'

# initialize PyAudio and open stream
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT, channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK,
    input_device_index=None,
)

# fast whisper model
whisper = WhisperModel('../models/faster-whisper-small', device='cpu', compute_type='int8')

# pyannote segmentation model
segmentation = Model.from_pretrained('../models/pyannote-segmentation-3/pytorch_model.bin')
pipeline = VoiceActivityDetection(segmentation=segmentation)
HYPER_PARAMETERS = {
    # remove speech regions shorter than that many seconds.
    'min_duration_on': 0.15,
    # fill non-speech regions shorter than that many seconds.
    'min_duration_off': 0.1,
}
pipeline.instantiate(HYPER_PARAMETERS)
max_silent_duration = 2.0


def record_audio():
    print('Recording started...')
    is_recording = True
    frames.clear()
    silent_duration = 0.0

    while is_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        waveform = torch.from_numpy(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
        waveform = waveform.unsqueeze(0)

        vad_result = pipeline({'waveform': waveform, 'sample_rate': RATE})

        print('VAD result: ', vad_result)
        print('VAD timeline: ', vad_result.get_timeline())

        if vad_result.get_timeline():
            silent_duration = 0.0
        else:
            silent_duration += CHUNK / RATE

        if silent_duration >= max_silent_duration:
            is_recording = False

    print('Recording stopped.')

    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    segments, info = whisper.transcribe('tmp.wav', beam_size=5, language='ko')
    user_input = ''
    for segment in segments:
        user_input += segment.text
    print(f"User input: {user_input}")


def main():
    root = tk.Tk()

    record_start = tk.Button(
        root, text='start recording',
        command=lambda: threading.Thread(target=record_audio).start(),
    )
    record_start.grid(row=0, column=1, pady=10)

    root.mainloop()


if __name__ == '__main__':
    main()
