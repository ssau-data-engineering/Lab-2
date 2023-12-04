import tensorflow as tf
import numpy as np
learning_rate = 1e-2
from mnist import load_dataset

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)
mean_image = np.mean(X_train, axis=0)

X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

# X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
# X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
# X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

def model_init_fn():
    input_shape = (784,1)
    hidden_layer_size, num_classes = 200, 10
    initializer = tf.initializers.VarianceScaling(scale=2.0,seed=42)
    layers = [
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu',
                              kernel_initializer=initializer),
        tf.keras.layers.Dense(num_classes, activation='softmax', 
                              kernel_initializer=initializer),
    ]
    model = tf.keras.Sequential(layers)
    return model

csv_logger = tf.keras.callbacks.CSVLogger('log.csv', append=True, separator=';')
model = model_init_fn()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val),callbacks=[csv_logger])
model.evaluate(X_test, y_test)
model.save('/data/')