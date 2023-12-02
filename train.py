import logging

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

logging.basicConfig(filename='training_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    df_train = pd.read_csv('boston_train_data.csv')

    train_data = df_train.drop('target', axis=1).values
    train_targets = df_train['target'].values

    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    class LogToFileCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            msg = f"Epoch {epoch+1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - mae: {logs['mae']:.4f} - val_loss: {logs['val_loss']:.4f} - val_mae: {logs['val_mae']:.4f}"
            logging.info(msg)

    history = model.fit(train_data, train_targets, epochs=100, batch_size=16, validation_split=0.2, verbose=2, callbacks=[LogToFileCallback()])
