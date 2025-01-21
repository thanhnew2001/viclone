import os
import torch
import torchaudio
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from underthesea import sent_tokenize
from datetime import datetime
from unidecode import unidecode
from vinorm import TTSnorm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Initialize Flask app
app = Flask(__name__)

# Directories for uploads and output
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Allowed file extensions for audio
ALLOWED_EXTENSIONS = {'wav'}

# Global variable to store the model instance
vixtts_model = None

# Initialize model once when the app starts
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

# Run TTS function
def run_tts(XTTS_MODEL, lang, tts_text, speaker_audio_file, normalize_text=True):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "Error with model or speaker audio", None

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )

    # Normalize text if needed
    if normalize_text and lang == "vi":
        tts_text = normalize_vietnamese_text(tts_text)

    tts_texts = sent_tokenize(tts_text)
    wav_chunks = []
    
    for text in tts_texts:
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

        wav_chunks.append(wav_chunk["wav"])

    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    out_path = os.path.join(output_dir, f"{get_file_name(tts_text)}.wav")
    torchaudio.save(out_path, out_wav, 24000)
    return out_path

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['input_text']
    language = request.form['language']
    normalize_text = 'normalize_text' in request.form

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

        if vixtts_model is None:
            # If model is not loaded, load it here
            global vixtts_model
            vixtts_model = load_model('model/model.pth', 'model/config.json', 'model/vocab.json')

        # Run TTS
        audio_file = run_tts(vixtts_model, language, input_text, file_path, normalize_text)

        # Return the URL for the generated audio file
        return jsonify({'audio_file': f'/download/{os.path.basename(audio_file)}'})

    return jsonify({'error': 'Invalid file'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load the model outside of API call (when the app starts)
    vixtts_model = load_model('model/model.pth', 'model/config.json', 'model/vocab.json')

    app.run(debug=True)
