import pandas as pd
import tensorflow as tf
import numpy as np

def load_mnist(num_training=49000, num_validation=1000, num_test=10000):
    mnist = tf.keras.datasets.mnist.load_data()
    (X_train, y_train), (X_test, y_test) = mnist
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
        
    X_train_csv = pd.DataFrame(X_train.reshape(-1, 28*28))
    X_train_csv.to_csv('/data/X_train.csv', index=False)
    y_train_csv = pd.DataFrame(y_train)
    y_train_csv.to_csv('/data/y_train.csv', index=False)
    
    X_val_csv = pd.DataFrame(X_val.reshape(-1, 28*28))
    X_val_csv.to_csv('/data/X_val.csv', index=False)
    y_val_csv = pd.DataFrame(y_val)
    y_val_csv.to_csv('/data/y_val.csv', index=False)

load_mnist()