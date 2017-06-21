import gzip
import pickle
import numpy as np
from model import model
from tflearn.data_utils import shuffle, to_categorical

# Data loading and preprocessing
f = gzip.open('planesnet.pklz', 'rb')
planesnet = pickle.load(f)
f.close()
X = planesnet['data'] / 255.
X = X.reshape([-1,3,20,20]).transpose([0,2,3,1])
Y = np.array(planesnet['labels'])
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 2)

# Train the model
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=.2,
          show_metric=True, batch_size=96, run_id='planesnet_cnn')

# Save model
model.save("model.tfl")