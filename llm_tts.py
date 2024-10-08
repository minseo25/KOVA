import os
import tkinter as tk

# LLM2TTSThread class and other helper functions
from LLM2TTSThread import LLM2TTSThread
from utils import record_audio, stt_process, get_clipboard_content

# for windows (if you have duplicate dll initialization error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# global variable for multithreading
thread = None


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
        thread = LLM2TTSThread(user_input, mime_type, content)
    else:
        thread = LLM2TTSThread(user_input, 'text/plain', '')
    thread.start()


def main():
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

    # Button to ask LLM
    ask_button = tk.Button(
        root, text='ask to llm',
        command=lambda: ask_llm(use_clipboard.get()),  # Pass the checkbox state to the function
    )
    ask_button.grid(row=4, column=1, pady=10)

    root.mainloop()


if __name__ == '__main__':
    main()
