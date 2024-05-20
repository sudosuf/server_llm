#Работает через powershell  распознает формат wav
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import pyaudio
import ffmpeg
import soundfile as sf
import shutup; shutup.please()
import time

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
time_start = time.time()
data, samplerate = sf.read('test_audio_neft_termin_(1).wav') # вытаскиваем данные из аудио файла
result = pipe(data) # передаем данные из аудио файла/ Если передовать напрямую аудиофайл то вылезит ошибка с распознаванием файла
time_end = time.time()
print(result["text"])
print(time_end-time_start)

