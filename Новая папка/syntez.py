from transformers import VitsModel, AutoTokenizer, set_seed
import torch
import scipy
from ruaccent import RUAccent
import winsound
def speechong(text_qe):
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

    # text
    text = """Отвечаю на ваш вопрос"""+ text_qe

    # the placement of accents
    text = accentizer.process_all(text)
    print(text)
    # н+очью дв+адцать тр+етьего и+юня н+ачал изверг+аться с+амый выс+окий
    # д+ействующий вулк+ан в евр+азии - ключевск+ой. об +этом сообщ+ила
    # руковод+итель камч+атской гр+уппы реаг+ирования на вулкан+ические
    # изверж+ения, вед+ущий на+учный сотр+удник инстит+ута вулканол+огии
    # и сейсмол+огии дво ран +ольга г+ирина. « зафикс+ированное н+очью не
    # пр+осто свеч+ение, а верш+инное эксплоз+ивное изверж+ение
    # стромболи+анского т+ипа. пок+а так+ое изверж+ение ником+у не оп+асно:
    # ни насел+ению, ни ави+ации » поясн+ила тасс госпож+а г+ирина.

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs.to(device), speaker_id=speaker).waveform
        output = output.detach().cpu().numpy()

    scipy.io.wavfile.write("tts_audio_1.wav", rate=model.config.sampling_rate, data=output[0])

    winsound.PlaySound("tts_audio.wav", winsound.SND_FILENAME)