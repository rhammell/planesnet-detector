"""
Train and export machine learning model using PlanesNet dataset
"""

import gzip
import pickle
import numpy as np
from tflearn.data_utils import to_categorical
from model import model

f = gzip.open('planesnet.pklz', 'rb')
planesnet = pickle.load(f)
f.close()

# Preprocess image data for input
X = planesnet['data'] / 255.
X = X.reshape([-1,3,20,20]).transpose([0,2,3,1])
Y = np.array(planesnet['labels'])
Y = to_categorical(Y, 2)

# Train the model
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=.2,
          show_metric=True, batch_size=128, run_id='planesnet')

# Save trained model
model.save("models/model.tfl")