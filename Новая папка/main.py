import torch
import asyncio
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
import winsound
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

from transformers import VitsModel, AutoTokenizer, set_seed
import torch
import scipy
from ruaccent import RUAccent
from syntez import speechong

device = 'cuda'  # 'cpu' or 'cuda'

speaker = 0  # 0-woman, 1-man

set_seed(555)  # make deterministic

# load model
model_name = "utrobinmv/tts_ru_free_hf_vits_low_multispeaker"

model = VitsModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# load accentizer
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo', use_dictionary=True, device=device)


#-------------------------------------------------------------------------------------------------------------------------------------------------


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


# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

def record_audio(): # Запись аудио для sql model
    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone(sample_rate=24000)
    while True:
        with microphone:
            recognition_data = ""
            recognizer.adjust_for_ambient_noise(microphone, duration=1)
            winsound.PlaySound("tts_audio.wav", winsound.SND_FILENAME)

            print("Алиса вас слушает")
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
            return result["text"]
        break


# time_start = time.time()
# data, samplerate = sf.read('Audio.wav')
# print(samplerate)# вытаскиваем данные из аудио файла mono 24kHz
# result = pipe(data, generate_kwargs={"language": "russian"})  # передаем данные из аудио файла/ Если передовать напрямую аудиофайл то вылезит ошибка с распознаванием файла
# time_end = time.time()
# print(result["text"])
# print(time_end - time_start)

def sql_coder(result):
    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    completion = client.chat.completions.create(
        model="RichardErkhov/defog_-_sqlcoder-7b-2-gguf/sqlcoder-7b-2.Q6_K.gguf",
        messages=[
            {"role": "system", "content": """create table user(
    id_user int(20) primary key;
    name str(20) ---- Имя сотрудника;
    surname str(20) ---- Фамилия сотрудника;
    age int(20) ---- Возраст сотрудника)
    
    create table work(
    id_job int(20) primary key;
    id_user int(20);
    name_company str(20) ---- Название компании )
    
     ---- connect table user t join table work v: t.id_user=v.id_user.  """},
            {"role": "user", "content": f'+{result}+'}
        ],
        temperature=0.7,
    )
    answer_sql = str(completion.choices[0].message.content)
    temp = answer_sql.replace("', role='assistant', function_call=None, tool_calls=None)", "")
    answer_sql = temp.replace("""ChatCompletionMessage(content= " """, "")
    print(answer_sql)



def listen_audio():
    while True:
        frequency = 24000

        # Recording duration in seconds
        duration = 3

        # to record audio from
        # sound-device into a Numpy
        print("shadow listen start")
        recording = sd.rec(int(duration * frequency),
                           samplerate=frequency, channels=1)

        # Wait for the audio to complete
        sd.wait()
        print("shadow listen end")

        # using scipy to save the recording in .wav format
        # This will convert the NumPy array
        # to an audio file with the given sampling frequency
        write("recording0.wav", frequency, recording)
        data, samplerate = sf.read('recording0.wav')

        result = pipe(data, generate_kwargs={"language": "russian"})
        print(result)
        txt = result["text"]
        print(txt)
        if "Алиса" in txt:
            print("true")
            print("Слушаю")
            recognition_rezult = record_audio()
            print(recognition_rezult)
            sql_coder(recognition_rezult)



if __name__ == '__main__':
    listen_audio()
