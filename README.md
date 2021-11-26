This repo is a re-implementation of the model published in One-Shot Texture Retrieval with Global Context Metric in ijcai2019.


## To run the training on your own pc:

Navigate to the folder directory, open a terminal and create a virtual environment:
```
python3 -m venv env               # Create a virtual environment

source env/bin/activate           # Activate virtual environment
```
Install none pytorch dependencies:
```
pip install -r requirements.txt   # Install dependencies
```
Install pytorch 1.10 (you might have to use a different version depending on your CUDA version)
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Install albumentations if you want to play with data augmentation:
```
pip install -U albumentations
```
To exit virtual environment:
```
deactivate  # Exit virtual environment
```
