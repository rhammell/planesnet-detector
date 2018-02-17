"""
Train and export machine learning model using PlanesNet dataset
"""

import sys
import json
import numpy as np
from tflearn.data_utils import to_categorical
from model import model

def train(fname, out_fname):
    """ Train and save CNN model on Planesnet dataset

    Args:
        fname (str): Path to PlanesNet JSON dataset
        out_fname (str): Path to output Tensorflow model file (.tfl)
    """

    # Load planesnet data
    f = open(fname)
    planesnet = json.load(f)
    f.close()

    # Preprocess image data and labels for input
    X = np.array(planesnet['data']) / 255.
    X = X.reshape([-1,3,20,20]).transpose([0,2,3,1])
    Y = np.array(planesnet['labels'])
    Y = to_categorical(Y, 2)

    # Train the model
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=.2,
              show_metric=True, batch_size=128, run_id='planesnet')

    # Save trained model
    model.save(out_fname)


# Main function
if __name__ == "__main__":

    # Train using input file
    train(sys.argv[1], sys.argv[2])
