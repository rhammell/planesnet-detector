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

[Additional Results](http://imgur.com/a/z34B3)
[Additional REsults](https://www.kaggle.com/rhammell/planesnet)
[PlanesNet](https://www.kaggle.com/rhammell/planesnet)

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
A convolutional neural network (CNN) is defined within the `model.py` module using the (TFLearn)[http://tflearn.org/] library. This model is designed for PlanesNet image dimesional input. 

## Training
A copy of the PlanesNet dataset is included as `planesnet.pklz`. For more information on the layout of this dataset, and to get the current version of this file, see the [PlanesNet](https://www.kaggle.com/rhammell/planesnet) documentation. 

The defined CNN can be trained by running `train.py`. 
```bash
# Train the model
python train.py 
```
Outputs the model parameters calculated from training are saved into the `models` directory. Pre-trained model files are made available in this directory already. This trained model has achieved a classification accurary of >99.5% on the PlanesNet dataset. 

## Detector
Using the trained model files, a sliding window detector function can be run on any input image using `detector.py`. 

```bash
# Run on demo image
python detector.py "images/scene_1.png"
```
