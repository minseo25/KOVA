import pyaudio
import wave
import tkinter as tk
from threading import Thread
from faster_whisper import WhisperModel
import os

# for windows (if you have duplicate dll initialization error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 녹음 파라미터 설정
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # 1 채널 (모노)
RATE = 44100              # 44.1kHz 샘플링 레이트
CHUNK = 2048              # 버퍼 크기 증가
OUTPUT_FILENAME = 'output.wav'  # 저장할 파일 이름

# PyAudio 객체 생성
audio = pyaudio.PyAudio()

# 스트림 열기
stream = audio.open(
    format=FORMAT, channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK,
    input_device_index=None,
)  # 입력 장치 인덱스 설정

frames = []
is_recording = True

model = WhisperModel('../models/faster-whisper-small', device='cpu', compute_type='int8')


def record_audio():
    global is_recording
    print('녹음 시작...')
    while is_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    print('녹음 완료.')

    # 스트림 종료
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # WAV 파일로 저장
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    segments, info = model.transcribe(OUTPUT_FILENAME, beam_size=5, language='ko')
    for segment in segments:
        print(f'[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}')


def stop_recording():
    global is_recording
    is_recording = False


# GUI 만들기
root = tk.Tk()
root.title('녹음 프로그램')

start_button = tk.Button(root, text='녹음 시작', command=lambda: Thread(target=record_audio).start())
start_button.pack(pady=20)

stop_button = tk.Button(root, text='녹음 종료', command=stop_recording)
stop_button.pack(pady=20)

root.mainloop()
