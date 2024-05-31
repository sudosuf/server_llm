import speech_recognition
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from speech_recognition import Recognizer

recognizer_pers = Recognizer()
recognizer_pers.pause_threshold = 10 # время паузы в произношении после которого ПО распознает окончание предложения.

################################################################################################# ДАННАЯ ФУНКЦИЯ ВЫЗЫВАЕТЬСЯ ПРИ НАЖАТИИ НА КНОПКУ АУДИОЗАПИСИ. ################################################
def record_audio():  # Запись аудио для нейроной сети ии последующее его распознавания
    """
    ДАННАЯ ФУНКЦИЯ СОЗДАНА ДЛЯ ЗАПИСИ ГОЛОСА СЛУШАТЕЛЯ, ПОКА ОН НЕ ПЕРЕСТАНЕТ ГОВОТЬ. А ПОСЛЕ ПОССЫЛКИ АУДИОДАННЫХ НА РАСПОЗНАВАНИИ МОДЕЛИ.
    :input:  NONE
    :return: TEXT
    """
    recognizers = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone(sample_rate=24000)
    with microphone:
        recognizers.adjust_for_ambient_noise(microphone, duration=1) # измерение уровня шума и адаптация чувствительности микрофона | duration -- время настройки микрофона
        print("Лука вас слушает")
        try:
            audio = recognizers.listen(microphone, 8, 29)
        except speech_recognition.WaitTimeoutError:
            return ("Время ожидания истекло. Попробуйте снова")
        except speech_recognition.UnknownValueError:
               return ("Не удалось распознать речь. Повторите попытку")
        except speech_recognition.RequestError:
             return ("Не удалось преобразовать текст")


        # with open('Audio.wav', 'wb') as file:
        #     wav_data = audio.get_wav_data()
        #     file.write(wav_data)
        # data, samplerate = sf.read('Audio.wav')# вытаскиваем данные из аудио файла mono 24kHz

        audio_wav = audio.get_wav_data() # Если данное решение не работает используйте код выше через запись аудио в файл и последующего извлечение данных из него
        result = pipe(audio_wav, generate_kwargs={"language": "russian"})  # Если audio_waw не работае, то замениет его на data из закоментированного выше кода
        print(result["text"])
        return result["text"]

############################################################################### ВСТАВЬТЕ ЭТОТ КОД В НАЧАЛО РАБОТЫ ПРИЛОЖЕНИЯ, ЧТОБЫ МОДЕЛЬ ПОДГРУЖАЛАСЬ ВМЕСТЕ С ЗАПУСКОМ ПРИЛОЖЕНИЯ, А НЕ ПРИ КАЖДОМ ОБРАЩЕНИИ К НЕЙ ########################################################################################

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 #if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

models = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
models.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=models,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
########################################################################################################################################################################################################################################################################################################################