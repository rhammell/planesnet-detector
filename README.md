# planesnet-detector
This repository contains script files for performing...

## Setup
These scripts make use of the lastest version of Tensorflow, which requires Python 3.5+

```bash
# Clone this repository
git clone https://github.com/rhammell/planesnet-detector.git

# Go into the repository
cd planesnet-detector

# Install required modules
pip install -r requirements.txt
```

## Model Training
A pre-trained TFLearn convolutional neural net (CNN) model is stored in the `models` folder. The design and parameters of this network can be seen in `model.py`. Retrain or save a new network by running `train.py`.   

```bash
# Train the model
python train.py 
```

## Sliding Window Detector

```bash
# Run on demo image
python detector.py "images/scene_1.png"
```
