from faster_whisper import WhisperModel
import datetime


# run on CPU with INT8
model = WhisperModel('../models/faster-whisper-small', device='cpu', compute_type='int8')

# check elapsed time for transcribing in ms
dt = datetime.datetime.now().microsecond
segments, info = model.transcribe('output.wav', beam_size=5, language='ko')
dt2 = datetime.datetime.now().microsecond
print('Elapsed time: %d ms' % ((dt2 - dt)/1000))

print(f"Detected language '{info.language}' with probability {info.language_probability:f}")

for segment in segments:
    print(f'[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}')
