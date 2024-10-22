import os
import threading
import time
import tkinter as tk
import sounddevice as sd
import soundfile as sf
from PIL import Image, ImageTk

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc

# LLM2TTSThread class
from LLM2TTSThread import LLM2TTSThread
from utils import record_audio, stt_process, get_clipboard_content

# for windows (if you have duplicate dll initialization error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# hotword detection using eff_word_net
base_model = Resnet50_Arc_loss()
malbud_hw = HotwordDetector(
    hotword='hi_kova',
    model=base_model,
    reference_file=os.path.join(samples_loc, 'hi_kova_ref.json'),
    threshold=0.65,
    relaxation_time=5,  # hotword 감지 후, 5초동안은 추가 감지 방지
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
gif_frames: list = []
img_label = None
root = None
frame_index = 1
gif_playing = True


def update_recording_status(status):
    global recording
    recording = status

    if recording:
        msg.set('Recording...')
        label.config(fg='red')
    else:
        msg.set('Ready to detect hotword.')
        label.config(fg='black')


def ask_llm():
    global thread, use_clipboard, freeze_until
    if thread is not None:
        thread.stop()
        # thread.join()  # this will block the main thread

    # update recording status in label
    update_recording_status(True)
    record_audio()
    update_recording_status(False)
    freeze_until = time.time() + 5.0  # freeze for 5 seconds

    # STT
    user_input = stt_process()

    if len(user_input) == 0:
        return
    print('User input:', user_input)

    if use_clipboard.get():
        mime_type, content = get_clipboard_content()
        thread = LLM2TTSThread(
            user_input, mime_type, content,
            start_animation_callback=start_animation, stop_animation_callback=stop_animation,
        )
    else:
        thread = LLM2TTSThread(
            user_input, 'text/plain', '',
            start_animation_callback=start_animation, stop_animation_callback=stop_animation,
        )
    thread.start()


def detect_hotword():
    global recording, freeze_until, mic_stream
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
            mic_stream.clear_stream()


def start_animation():
    global gif_playing
    gif_playing = True
    update_gif()


def stop_animation():
    global gif_playing, frame_index, img_label
    gif_playing = False
    frame_index = 1
    img_label.configure(image=gif_frames[0])


def update_gif():
    global gif_frames, img_label, frame_index
    if not gif_playing:
        return

    frame = gif_frames[frame_index]

    img_label.config(image=frame)
    img_label.image = frame  # 참조 유지

    frame_index = (frame_index + 1) % len(gif_frames)
    root.after(50, update_gif)


def main():
    global msg, label, use_clipboard, root, gif_frames, img_label

    root = tk.Tk()
    root.title('Ask to MAL-BUD!')

    # load gif frames
    gif = Image.open('./assets/soundwave.gif')

    for idx in range(gif.n_frames):
        gif.seek(idx)
        frame = ImageTk.PhotoImage(gif)
        gif_frames.append(frame)
    for idx in range(gif.n_frames - 1, -1, -1):
        gif.seek(idx)
        frame = ImageTk.PhotoImage(gif)
        gif_frames.append(frame)

    # Label for the gif
    img_label = tk.Label(root, image=gif_frames[0])
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
