import numpy as np
from PIL import Image
from model import model

# Load in trained model
model.load("model.tfl")

# Read input image data
im = Image.open("scene1.png")
arr = np.array(im)[:,:,0:3]
shape = arr.shape

# Create output image
output = np.copy(arr)

# Set sliding window params
step = 30
win = 20

# Loop through image 
for i in range(0, shape[0]-win, step):
    print(i)
    for j in range(0, shape[1]-win, step):

        # Extract chip
        chip = arr[i:i+win,j:j+win,:]
        
        # Make ML prediction
        prediction = np.argmax(model.predict([chip / 255.])[0])
        
        # Draw bounding box around positive detections
        if prediction == 1:
            output[i:i+win, j, 0:3] = [255,0,0]
            output[i:i+win, j+win-1, 0:3] = [255,0,0]
            output[i, j:j+win, 0:3] = [255,0,0]
            output[i+win-1, j:j+win, 0:3] = [255,0,0]

# Save output image
outIm = Image.fromarray(output)
outIm.save("scene1_detections3.png")