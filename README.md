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
