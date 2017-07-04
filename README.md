# planesnet-detector
This repository contains scripts that enable the detection of aircraft in [Planet](https://www.planet.com/) imagery using machine learning techniques. Included are files which define a machine learning model, train it using the [PlanesNet](https://www.kaggle.com/rhammell/planesnet) dataset, and apply it across an entire image scene to highlight aircraft detections.

## Methodology
[PlanesNet](https://www.kaggle.com/rhammell/planesnet) is a labeled training dataset consiting of image chips extracted from Planet satellite imagery. It contains thousands of 20x20 pixel RGB image chips labeled with either a "plane" or "no-plane" classification. Machine learning models can be trained against this data to classify any given input chip into either one of these classes. 

With an accurately trained model, this classification process can be extended to a full Planet image scene by using a sliding window technique. A 20x20 pixel window is moved across each pixel position in the image, extracted, and classified by the model. Neighboring pixel poistions that are classified as "plane" are then clustered into single detections. These detections are highlighted in an output image by drawing a bounding box around them in an output copy of the original Planet scene. 

See an example of the results below. 
<p>
<img src="http://i.imgur.com/2a6E9Nj.png" width="400">
<img src="http://imgur.com/d50SQA3.png" width="400">
</p>

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

## Model

## Training

## Detector

## Model Training
The PlanesNet labeled training dataset can be used at input to train machine learning models to classify a single 20x20 pixel image chip as either belonging to a 'plane' or 'no-plane' class. 

The PlanesNet labeled training dataset contains 20x20 pixel image chips labeled as belonging to either a 'plane' or 'no-plane' class, depending on the content of the image chp

PlanesNet is a labeled training dataset consiting of image chips extracted from Planet satellite imagery. It contains thousands of 20x20 pixel RGB images labeled with either a "plane" or "no-plane" classification. Machine learning models can be trained using this data to classify any 20x20 pixel input image as being one of those classes, depending on the input image content. 

The purpose of PlanesNet is to serve as labeled training data to train machine learning algorithms to detect the locations of airplanes in Planet's medium resolution remote sensing imagery.

The dataset includes 24000 20x20 RGB images labeled with either a "plane" or "no-plane" classification. Image chips were derived from PlanetScope full-frame visual scene products, which are orthorectified to a 3 meter pixel size.

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
