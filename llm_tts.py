import os
import tkinter as tk
from PIL import Image, ImageTk

# LLM2TTSThread class and other helper functions
from LLM2TTSThread import LLM2TTSThread
from utils import record_audio, stt_process, get_clipboard_content

# for windows (if you have duplicate dll initialization error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# global variables
thread = None
gif_frames: list = []
img_label = None
root = None
frame_index = 1
gif_playing = True

def start_animation():
    global gif_playing
    gif_playing = True
    animate_gif()

def stop_animation():
    global gif_playing, frame_index, img_label
    gif_playing = False
    frame_index = 1
    img_label.configure(image=gif_frames[0])

def animate_gif():
    global frame_index, gif_playing
    if gif_playing:
        frame = gif_frames[frame_index]
        img_label.configure(image=frame)
        frame_index = (frame_index + 1) % len(gif_frames)
        root.after(50, animate_gif)

def ask_llm(use_clipboard):
    global thread
    if thread is not None:
        thread.stop()
        # thread.join()  # this will block the main thread

    record_audio()
    # STT
    user_input = stt_process()

    if len(user_input) == 0:
        return

    if use_clipboard:
        mime_type, content = get_clipboard_content()
        thread = LLM2TTSThread(user_input, mime_type, content, start_animation_callback=start_animation, stop_animation_callback=stop_animation)
    else:
        thread = LLM2TTSThread(user_input, 'text/plain', '', start_animation_callback=start_animation, stop_animation_callback=stop_animation)
    thread.start()

def main():
    global gif_frames, img_label, root
    root = tk.Tk()
    root.title('Ask to MAL-BUD!')

    # load gif frames
    gif = Image.open("./assets/soundwave.gif")
    for frame in range(0, gif.n_frames):
        gif.seek(frame)
        frame_image = ImageTk.PhotoImage(gif.copy())
        gif_frames.append(frame_image)

    img_label = tk.Label(root, image=gif_frames[0])
    img_label.grid(row=0, column=0, columnspan=3, pady=10, padx=10)

    # Label for the checkbox
    clipboard_label = tk.Label(root, text='클립보드 내용도 전송')
    clipboard_label.grid(row=2, column=1)

    # Checkbox for clipboard content
    use_clipboard = tk.BooleanVar()  # Holds the state of the checkbox (True/False)
    clipboard_checkbox = tk.Checkbutton(root, variable=use_clipboard)
    clipboard_checkbox.grid(row=3, column=1)

    # Button to ask LLM
    ask_button = tk.Button(
        root, text='ask to llm',
        command=lambda: ask_llm(use_clipboard.get()),  # Pass the checkbox state to the function
    )
    ask_button.grid(row=4, column=1, pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()
