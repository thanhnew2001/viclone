import os
import string
import unicodedata
from datetime import datetime
from pprint import pprint

from tqdm import tqdm
import torch
import torchaudio
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from underthesea import sent_tokenize
from unidecode import unidecode
from vinorm import TTSnorm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Initialize Flask app
app = Flask(__name__)

# Directories for uploads and output
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
SAMPLES_FOLDER = 'static/samples'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SAMPLES_FOLDER'] = SAMPLES_FOLDER

# Allowed file extensions for audio
ALLOWED_EXTENSIONS = {'wav'}

# Global variable to store the model instance
vixtts_model = None

# Initialize model once when the app starts
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    XTTS_MODEL.load_checkpoint(config,
                               checkpoint_path=xtts_checkpoint,
                               vocab_path=xtts_vocab,
                               use_deepspeed=True)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    return XTTS_MODEL

# Normalize Vietnamese text
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
# Generate a safe filename based on input text
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

def run_tts(XTTS_MODEL, lang, tts_text, speaker_audio_file,
            normalize_text= True,
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
            print("Error normalize")

    if lang in ["ja", "zh-cn"]:
        tts_texts = tts_text.split("。")
    else:
        tts_texts = sent_tokenize(tts_text)

    if verbose:
        print("Text for TTS:")
        print(tts_texts)

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

# Flask routes
@app.route('/')
def index():
    samples = os.listdir(app.config['SAMPLES_FOLDER'])
    return render_template('index.html', samples=samples)

@app.route('/process', methods=['POST'])
def process():
    global vixtts_model  # Declare the variable as global so we can modify it
    
    input_text = request.form['input_text']
    language = request.form['language']
    normalize_text = 'normalize_text' in request.form

    # Handle sample selection
    sample_audio = request.form.get('sample_audio')
    if sample_audio:
        file_path = os.path.join(app.config['SAMPLES_FOLDER'], sample_audio)
    else:
        # Handle file upload
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            return jsonify({'error': 'Invalid file'}), 400

    if vixtts_model is None:
        # If model is not loaded, load it here
        vixtts_model = load_model('model/model.pth', 'model/config.json', 'model/vocab.json')

    # Run TTS
    audio_file = run_tts(vixtts_model,  language_code_map[language], input_text, file_path, normalize_text)

    # Return the URL for the generated audio file
    return jsonify({'audio_file': f'/download/{os.path.basename(audio_file)}'})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load the model outside of API call (when the app starts)
    vixtts_model = load_model('model/model.pth', 'model/config.json', 'model/vocab.json')

    app.run(debug=True)
