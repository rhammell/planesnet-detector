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
The repository contains a pre-trained convolutional neural net (CNN) model trained against the planesnet dataset and stored in the models folder.  

```bash
# Train the model
python train.py 
```

## Sliding Window Detector

```bash
# Run on demo image
python detector.py "images/scene_1.png"
```
