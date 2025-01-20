import os
import subprocess
import gradio as gr

def setup_environment():
    os.system("rm /etc/localtime")
    os.system("ln -s /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime")
    os.system("date")

    print(" > Cài đặt thư viện...")
    os.system("rm -rf TTS/")
    os.system("git clone --branch add-vietnamese-xtts -q https://github.com/thinhlpg/TTS.git")
    os.system("pip install --use-deprecated=legacy-resolver -q -e TTS")
    os.system("pip install deepspeed -q")
    os.system("pip install -q vinorm==2.0.7")
    os.system("pip install -q cutlet")
    os.system("pip install -q unidic==1.1.0")
    os.system("pip install -q underthesea")
    os.system("pip install -q gradio==4.35")
    os.system("pip install deepfilternet==0.5.6 -q")

    from huggingface_hub import snapshot_download
    os.system("python -m unidic download")
    
    print(" > Tải mô hình...")
    snapshot_download(repo_id="thinhlpg/viXTTS", repo_type="model", local_dir="model")

    print(" > ✅ Cài đặt hoàn tất, bạn hãy chạy tiếp các bước tiếp theo nhé!")

language = "Tiếng Việt"
input_text = "Sau khi cãi nhau ầm ĩ, vợ đòi chia tay, ông chồng nghe thế liền cầu cứu con gái 5 tuổi:  - Con gái, nói giúp bố đi! Mẹ con đòi ly hôn với bố kìa! Cô con gái dửng dưng đáp:  - Ly hôn thì ly hôn thôi! - Bố mẹ ly hôn mà con không quan tâm gì à? - ông bố mếu máo. - Cả việc lấy nhau mà bố mẹ còn không thèm hỏi ý kiến con. - cô con gái giận dỗi nói - Thì tại sao con phải quan tâm chuyện hai người chia tay chứ?  - !?!"
reference_audio = "model/user_sample.wav"
normalize_text = True
verbose = True
output_chunks = False

def cry_and_quit():
    print("> Lỗi rồi huhu 😭😭, bạn hãy nhấn chạy lại phần này nhé!")
    quit()

import string
import unicodedata
from datetime import datetime
from pprint import pprint

import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize
from unidecode import unidecode

try:
    from vinorm import TTSnorm
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except:
    cry_and_quit()

# Load model
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    
    print("Loading XTTS model!")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=True)
    
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return XTTS_MODEL

def get_file_name(text, max_char=50):
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    filename = f"{current_datetime}_{filename}"
    return filename

def calculate_keep_len(text, lang):
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = (
        text.count(".")
        + text.count("!")
        + text.count("?")
        + text.count(",")
    )

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1

def normalize_vietnamese_text(text):
    text = (
        TTSnorm(text, unknown=False, lower=False, rule=True)
        .replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
    )
    return text

def run_tts(XTTS_MODEL, lang, tts_text, speaker_audio_file,
            normalize_text=True,
            verbose=False,
            output_chunks=False):
    """
    Run text-to-speech (TTS) synthesis using the provided XTTS_MODEL.

    Args:
        XTTS_MODEL: A pre-trained TTS model.
        lang (str): The language of the input text.
        tts_text (str): The text to be synthesized into speech.
        speaker_audio_file (str): Path to the audio file of the speaker to condition the synthesis on.
        normalize_text (bool, optional): Whether to normalize the input text. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Defaults to False.
        output_chunks (bool, optional): Whether to save synthesized speech chunks separately. Defaults to False.

    Returns:
        str: Path to the synthesized audio file.
    """

    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )

    if normalize_text and lang == "vi":
        # Bug on google colab
        try:
            tts_text = normalize_vietnamese_text(tts_text)
        except:
            cry_and_quit()

    if lang in ["ja", "zh-cn"]:
        tts_texts = tts_text.split("。")
    else:
        tts_texts = sent_tokenize(tts_text)

    if verbose:
        print("Text for TTS:")
        pprint(tts_texts)

    wav_chunks = []
    for text in tqdm(tts_texts):
        if text.strip() == "":
            continue

        wav_chunk = XTTS_MODEL.inference(
            text=text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
        )

        # Quick hack for short sentences
        keep_len = calculate_keep_len(text, lang)
        wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len])

        if output_chunks:
            out_path = os.path.join(output_dir, f"{get_file_name(text)}.wav")
            torchaudio.save(out_path, wav_chunk["wav"].unsqueeze(0), 24000)
            if verbose:
                print(f"Saved chunk to {out_path}")

        wav_chunks.append(wav_chunk["wav"])

    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    out_path = os.path.join(output_dir, f"{get_file_name(tts_text)}.wav")
    torchaudio.save(out_path, out_wav, 24000)

    if verbose:
        print(f"Saved final file to {out_path}")

    return out_path

language_code_map = {
    "Tiếng Việt": "vi",
    "Tiếng Anh": "en",
    "Tiếng Tây Ban Nha": "es",
    "Tiếng Pháp": "fr",
    "Tiếng Đức": "de",
    "Tiếng Ý": "it",
    "Tiếng Bồ Đào Nha": "pt",
    "Tiếng Ba Lan": "pl",
    "Tiếng Thổ Nhĩ Kỳ": "tr",
    "Tiếng Nga": "ru",
    "Tiếng Hà Lan": "nl",
    "Tiếng Séc": "cs",
    "Tiếng Ả Rập": "ar",
    "Tiếng Trung (giản thể)": "zh-cn",
    "Tiếng Nhật": "ja",
    "Tiếng Hungary": "hu",
    "Tiếng Hàn": "ko",
    "Tiếng Hindi": "hi"
}

def tts_interface(input_text, reference_audio, normalize_text, verbose, output_chunks):
    if not os.path.exists(reference_audio):
        return "Bạn chưa tải file âm thanh lên. Hãy chọn giọng khác, hoặc tải file của bạn lên ở bên dưới.⚠️⚠️⚠️"
    else:
        audio_file = run_tts(vixtts_model,
                             lang=language_code_map[language],
                             tts_text=input_text,
                             speaker_audio_file=reference_audio,
                             normalize_text=normalize_text,
                             verbose=verbose,
                             output_chunks=output_chunks)
        return audio_file

# Gradio interface
iface = gr.Interface(
    fn=tts_interface,
    inputs=[
        gr.inputs.Textbox(lines=5, label="Văn bản để đọc"),
        gr.inputs.Textbox(default="model/user_sample.wav", label="Đường dẫn đến file âm thanh mẫu"),
        gr.inputs.Checkbox(default=True, label="Tự động chuẩn hóa chữ"),
        gr.inputs.Checkbox(default=True, label="In chi tiết xử lý"),
        gr.inputs.Checkbox(default=False, label="Lưu từng câu thành file riêng lẻ")
    ],
    outputs=gr.outputs.Audio(label="Kết quả âm thanh"),
    title="Text-to-Speech Demo",
    description="Nhập văn bản và nhấn nút để chuyển đổi thành âm thanh."
)

if __name__ == "__main__":
    print("> Đang nạp mô hình...")

    try:
        if not vixtts_model:
            vixtts_model = load_model(xtts_checkpoint="model/model.pth",
                                      xtts_config="model/config.json",
                                      xtts_vocab="model/vocab.json")
    except:
        vixtts_model = load_model(xtts_checkpoint="model/model.pth",
                                   xtts_config="model/config.json",
                                   xtts_vocab="model/vocab.json")

    print("> Đã nạp mô hình")
    iface.launch()
