import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import speech_recognition
import pyaudio
from pydub import AudioSegment
from ffmpeg import FFmpeg
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import time
import numpy as np
from openai import OpenAI

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

#dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
#sample = dataset[0]["audio"]

recognizer = speech_recognition.Recognizer()
microphone = speech_recognition.Microphone(sample_rate=24000)
while True:
    with microphone:
        recognition_data = ""
        recognizer.adjust_for_ambient_noise(microphone, duration=1)
        print("start")
        audio = recognizer.listen(microphone, 5, 29)
        # recording = sd.rec(int(5 * 24000), samplerate=24000, channels=1)
        # sd.wait()
        # arr = np.array(audio)
        #  write("Audio.wav", 24000, arr)
        with open('Audio.wav', 'wb') as file:
            wav_data = audio.get_wav_data()
            file.write(wav_data)

    data, samplerate = sf.read('Audio.wav')
    print(samplerate)  # вытаскиваем данные из аудио файла mono 24kHz
    result = pipe(data, generate_kwargs={"language": "russian"})
    print(result)
