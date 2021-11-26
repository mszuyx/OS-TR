This repo is a re-implementation of the model published in **One-Shot Texture Retrieval with Global Context Metric** in ijcai2019.
### Note that these code contain some of my own modifications / improvements, please do not confuse this repo with the original / official method!

## To run model training on your own pc:

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
To start the training:
```
python3 train_ver2.py
```
**If you encounter some runtime memory issues, you can decrease the batch_size / num_workers according to your GPU spec**

To test the trained model:
```
python3 localImageTest.py
```
**Most likely you will be getting "file not found error" since I hardcored most of the file directories, remember to change them in the scripts according to your file system.**

To exit virtual environment:
```
deactivate                       # Exit virtual environment
```

## DTD dataset can be downloaded from here:
https://www.robots.ox.ac.uk/~vgg/data/dtd/

Please place the downloaded .pth file under the root directory (where train.py is placed) for the train.py to work.

## The encoder backbone (pre-trained ResNet50 19c8e357) can be downloaded from here:
https://download.pytorch.org/models/resnet50-19c8e357.pth

Please place the downloaded .pth file under /utils/model/ for the train.py to work.
