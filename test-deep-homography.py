import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, Activation, Lambda
from keras.optimizers import SGD
from keras import backend as K
import glob
import numpy as np
import os.path
import matplotlib.pyplot as plt
import cv2


## Test the model
def test_model(model_save_path, test_data_path, test_size=2, batch_size=64):
    i = 0
    j = 0
    error = np.empty(int(test_size/batch_size)+1)
    images = np.empty((batch_size, 128, 128, 2))
    offsets = np.empty((batch_size, 8))
    model_l = load_model(model_save_path)
    for npz_test in glob.glob(os.path.join(test_data_path, '*.npz')):
        archive = np.load(npz_test)
        images[i] = np.resize(archive['images'], (128, 128, 2)).astype(np.float64)/255
        offsets[i] = archive['offsets']
        cv2.imwrite('image_{}_0.png'.format(i), images[i][:, :, 0]*255)
        cv2.imwrite('image_{}_1.png'.format(i), images[i][:, :, 1]*255)
        i = i + 1
        if i % batch_size == 0:
            offsets_predicted = model_l.predict(images)*64
            print(offsets_predicted.tolist())
            print(offsets.tolist())
            x_1 = np.sqrt((offsets[:, 0] - offsets_predicted[:, 0]) ** 2 + (offsets[:, 1] - offsets_predicted[:, 1]) ** 2)
            x_2 = np.sqrt((offsets[:, 2] - offsets_predicted[:, 2]) ** 2 + (offsets[:, 3] - offsets_predicted[:, 3]) ** 2)
            x_3 = np.sqrt((offsets[:, 4] - offsets_predicted[:, 4]) ** 2 + (offsets[:, 5] - offsets_predicted[:, 5]) ** 2)
            x_4 = np.sqrt((offsets[:, 6] - offsets_predicted[:, 6]) ** 2 + (offsets[:, 7] - offsets_predicted[:, 7]) ** 2)
            error[j] = np.average([x_1, x_2, x_3, x_4])
            print('Mean Corner Error: ', error[j])
            j = j + 1
            i = 0
    print('Mean Average Corner Error: ', np.average(error))

regression_file = 'models/regression.h5'
classification_file = 'models/classification.h5'

print("Testing the Regression Network...")
test_model(model_save_path='models/regression.h5', test_data_path='test-data/')

print("Testing the Classification Network...")
test_model(model_save_path='models/classification.h5', test_data_path='test-data/')
