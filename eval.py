import tensorflow as tf
import numpy as np
import os
import cv2
from nets import models_factory
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics

# Load model and checkpoint
def load_model(checkpoint_path, model_name='predrnn_pp'):
    model = models_factory.construct_model(model_name)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        print("Model restored from", checkpoint_path)
        return model, sess

# Prepare test data
def prepare_test_data(data_path):
    data = np.load(data_path)
    return data['inputs'], data['targets']

# Evaluate model
def evaluate_model(model, sess, test_inputs, test_targets):
    predictions = model.predict(test_inputs)
    mse = metrics.batch_mse(predictions, test_targets)
    ssim = metrics.batch_ssim(predictions, test_targets)
    print("MSE:", mse)
    print("SSIM:", ssim)
    return predictions, mse, ssim

if __name__ == '__main__':
    checkpoint_path = '/path/to/your/checkpoint'
    test_data_path = '/path/to/your/test_data.npz'
    
    model, sess = load_model(checkpoint_path)
    test_inputs, test_targets = prepare_test_data(test_data_path)
    predictions, mse, ssim = evaluate_model(model, sess, test_inputs, test_targets)
