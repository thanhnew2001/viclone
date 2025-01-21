import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from vinorm import TTSnorm

# Flask app setup
app = Flask(__name__)

# Path configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'wav'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model function (same as in your original code)
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
    XTTS_MODEL.load_checkpoint(config,
                               checkpoint_path=xtts_checkpoint,
                               vocab_path=xtts_vocab,
                               use_deepspeed=True)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    return XTTS_MODEL

# Run TTS function (same as in your original code)
def run_tts(XTTS_MODEL, lang, tts_text, speaker_audio_file, normalize_text=True):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "Error with model or speaker audio", None

    # Generate TTS (your logic here)
    # ...

    # Save and return the path to the audio file
    out_path = os.path.join(OUTPUT_FOLDER, "output.wav")
    torchaudio.save(out_path, torch.randn(1, 24000), 24000)  # Dummy output for now
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
        return 'No file part', 400
    file = request.files['audio_file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load model (adjust paths accordingly)
        xtts_model = load_model('model/model.pth', 'model/config.json', 'model/vocab.json')

        # Run TTS
        audio_file = run_tts(xtts_model, language, input_text, file_path, normalize_text)

        # Return result to the frontend
        return render_template('index.html', audio_file=f'/download/{os.path.basename(audio_file)}')
    return 'Invalid file', 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
