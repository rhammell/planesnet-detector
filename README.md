# planesnet-detector
This repository contains scripts that enable the detection of aircraft in [Planet](https://www.planet.com/) imagery using machine learning techniques. Included are files which define a machine learning model, train it using the [PlanesNet](https://www.kaggle.com/rhammell/planesnet) labeled dataset, and apply it across an entire image scene to highlight aircraft detections.

## Methodology



## Setup
Python 3.5+ is required for compatability with all required modules

```bash
# Clone this repository
git clone https://github.com/rhammell/planesnet-detector.git

# Go into the repository
cd planesnet-detector

# Install required modules
pip install -r requirements.txt
```

## Model Training
The PlanesNet dataset can be used to train machine learning models to classify a single 20x20 pixel image chip as either belonging to a 'plane' or 'no-plane' class depending on the content of the chip. 

A convolutional neural network (CNN) is defined within the `model.py` module using the (TFLearn)[http://tflearn.org/] library. 

Pre-trained model files are stored in the `models` directory. Train the network by running `train.py`.

This model is trained using the [PlanesNet](https://www.kaggle.com/rhammell/planesnet) dataset. This model has acheived an accuracy of >99.5% in classifying the 'plane' and 'no-plane' classes of the dataset.  


A TFLearn convulutional neural net (CNN) model designed to work with PlanesNet input is defined within the `model.py` module. Pre-trained model files are stored in the `models` folder. Retrain or save a new network by running `train.py`. 

```bash
# Train the model
python train.py 
```

## Sliding Window Detector
Using the trained model files, a sliding window detector function can be 

```bash
# Run on demo image
python detector.py "images/scene_1.png"
```

## Results
