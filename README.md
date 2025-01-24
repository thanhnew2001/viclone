# Clone and install:
git clone --branch add-vietnamese-xtts -q https://github.com/thinhlpg/TTS.git \
pip install --use-deprecated=legacy-resolver -q -e TTS \
pip install deepspeed -q \
pip install -q vinorm==2.0.7 \
pip install -q cutlet \
pip install -q unidic==1.1.0 \
pip install -q underthesea \
pip install -q gradio==4.35 \
pip install deepfilternet==0.5.6 -q \

# Download model: 
$python3 downloadmodel.py

# Run a test tts:
$python3 loadmodelandtest.py


# Install key
 export GOOGLE_APPLICATION_CREDENTIALS="key/inifiniti-5c931-firebase-adminsdk-fbsvc-6b397f960a.json"

# Start gunicorn (multi threds)
 nohup gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 --timeout 120 app:app &
 