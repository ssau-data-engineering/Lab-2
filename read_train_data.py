import pandas as pd
import tensorflow as tf
import numpy as np

csv_logger = [tf.keras.callbacks.CSVLogger('/data/log.csv', append=True, separator=';')]

def read_train_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_test = X_test[:400]
    y_test = y_test[:400]
    X_train = X_train / 255
    X_test = X_test / 255
    X_train_flat = X_train.reshape(len(X_train), (28 * 28))
    X_test_flat = X_test.reshape(len(X_test), (28 * 28))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape = (784,), activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'sigmoid'),
        tf.keras.layers.Dense(32, activation = 'sigmoid'),
        tf.keras.layers.Dense(10, activation = 'softmax'),
    ])

    model.compile(
        optimizer = 'adam',
        loss = "sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    
    logs = model.fit(X_train_flat, y_train, epochs=16, batch_size=1, verbose=2, callbacks=csv_logger)
    model.evaluate(X_test_flat, y_test)
read_train_data()