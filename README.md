# planesnet-detector
This repository contains scripts that enable the automatic detection of aircraft in [Planet](https://www.planet.com/) imagery using machine learning techniques. Included are files which define a machine learning model, train it using the Planesnet dataset, and apply it across an entire image scene to highlight aircraft detections.

## Methodology
[PlanesNet](https://www.kaggle.com/rhammell/planesnet) is a labeled training dataset consiting of image chips extracted from Planet satellite imagery. It contains thousands of 20x20 pixel RGB image chips labeled with either a "plane" or "no-plane" classification. Machine learning models can be trained against this data to classify any given input chip into either one of these classes. 

With an accurately trained model, this classification process can be extended to a full Planet image scene by using a sliding window technique. A 20x20 pixel window is moved across each pixel position in the image, extracted, and classified by the model. Neighboring window poistions that are classified as "plane" are then clustered into a single detection. These detections are highlighted with a bounding box in a copy of the original Planet scene.

See an example of the results below. 
<p>
<img src="https://i.imgur.com/imshZn6.png" width="400">
<img src="https://i.imgur.com/Fbzedgs.png" width="400">
</p>

[Additional Results](https://imgur.com/a/vYnQw)

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
A convolutional neural network (CNN) is defined within the `model.py` module using the [TFLearn](http://tflearn.org/) library. This model supports the 20x20x3 input dimensions of the PlanesNet image data.

## Training
The defined CNN can be trained with the JSON version of the PlanesNet dataset and saved to a Tensorflow .tfl file for later use. Train the model by running `train.py` and passing the path to `planesnet.json` and the path to the output .tfl file as arguments.

```bash
# Train the model
mkdir models
python train.py "planesnet.json" "models/model.tfl"
```

The latest version of `planesnet.json` is available through the [PlanesNet](https://www.kaggle.com/rhammell/planesnet) Kaggle page, which has further information describing the dataset layout. 

## Detector
A trained model can be applied across entire images using the sliding window detector function `detector.py`, which takes the model file path, input image path, and optional output image path as arguments. The output image will cluster positive detections and draw a bounding box around their center point. 

Example images are contained in the `images` directory. 
```bash
# Run on demo image with default output path
python detector.py "models/model.tfl" "images/scene_1.png"

# Run on demo image with defined output path
python detector.py "models/model.tfl" "image/scene_1.png" "image/scene_1_detections.png"
```
