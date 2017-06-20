import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from PIL import Image

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, 20, 20, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Load trained model
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load("model.tfl")

# Read input image data
im = Image.open("scene1.png")
arr = np.array(im)[:,:,0:3]
shape = arr.shape

# Create output image
output = np.copy(arr)

# Loop through image 
step = 3
sz = 20
for i in range(0, shape[0]-sz, step):
    print(i)
    for j in range(0, shape[1]-sz, step):

        # Extract chip
        chip = arr[i:i+sz,j:j+sz,:]
        
        # Make ML prediction
        prediction = np.argmax(model.predict([chip / 255.])[0])
        
        # Draw bounding box around detected chips
        if prediction == 1:
            output[i:i+sz-1,j,0:3] = [255,0,0]
            output[i:i+sz-1,j+sz-1,0:3] = [255,0,0]
            output[i,j:j+sz-1,0:3] = [255,0,0]
            output[i+sz-1,j:j+sz-1,0:3] = [255,0,0]

# Save output image
outIm = Image.fromarray(output)
outIm.save("scene1_detections.png")