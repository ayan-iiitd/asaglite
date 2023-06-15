# ASAGLite

This repository contains the source for the Python implementation of the paper ASAGLite: A Lightweight and Efficient Technique for Automatic Short Answer Grading.

##### To Generate Model
1. Create a python/jupyter environment
2. Activate the environment
3. Install dependencies
    1. `pip install pandas`
    2. `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
    3. `pip install -U sentence-transformers`
    4. `pip install --user -U nltk`
    5. `python -m nltk.downloader popular`
4. Define the configuration in `config.ini`
5. Train the model using `python train.py`
