import pandas as pd
import tensorflow as tf
import numpy as np

class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):

        assert X.shape[0] == y.shape[0]
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))

def train_part(model_init_fn, optimizer_init_fn, num_epochs=1, is_training=False):

    with tf.device('/device:GPU:0'):

        text_file = open("/data/train.txt", "w+")
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        model = model_init_fn()
        optimizer = optimizer_init_fn()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        t = 0
        for epoch in range(num_epochs):

            # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
            train_loss.reset_states()
            train_accuracy.reset_states()

            for x_np, y_np in train_dset:
                with tf.GradientTape() as tape:

                    # Use the model function to build the forward pass.
                    scores = model(x_np, training=is_training)
                    loss = loss_fn(y_np, scores)

                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    # Update the metrics
                    train_loss.update_state(loss)
                    train_accuracy.update_state(y_np, scores)

                    if t % print_every == 0:
                        val_loss.reset_states()
                        val_accuracy.reset_states()
                        for test_x, test_y in val_dset:
                            # During validation at end of epoch, training set to False
                            prediction = model(test_x, training=False)
                            t_loss = loss_fn(test_y, prediction)

                            val_loss.update_state(t_loss)
                            val_accuracy.update_state(test_y, prediction)

                        template = 'Iteration {}, Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {} \n'
                        
                        text_file.write(template.format(t, epoch+1,
                                             train_loss.result(),
                                             train_accuracy.result()*100,
                                             val_loss.result(),
                                             val_accuracy.result()*100))
                    t += 1
        text_file.close()

class CustomConvNet_B_D(tf.keras.Model):
    def __init__(self):
        super(CustomConvNet_B_D, self).__init__()
        channel_1, channel_2, num_classes = 28, 14, 10
        dp_rate = 0.2
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        self.conv1 = tf.keras.layers.Conv2D(channel_1, [3,3], [1,1], padding='same',
                                  kernel_initializer=initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.dp1 = tf.keras.layers.Dropout(rate=dp_rate)
        self.conv2 = tf.keras.layers.Conv2D(channel_2, [3,3], [1,1], padding='same',
                                  kernel_initializer=initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.dp2 = tf.keras.layers.Dropout(rate=dp_rate)
        self.fl = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes,
                                  activation='softmax',
                                  kernel_initializer=initializer)

    def call(self, input_tensor, training=False):
        x = input_tensor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dp2(x)

        x = self.fl(x)
        x = self.fc(x)
        return x
    

def model_init_fn():
    return CustomConvNet_B_D()

def optimizer_init_fn():
    learning_rate = 1e-3
    return tf.keras.optimizers.Adam(learning_rate)

print_every = 100
num_epochs = 10

X_train = np.asarray(pd.read_csv(f"/data/X_train.csv"), dtype=np.float32)
X_train = X_train.reshape(-1, 28, 28, 1)
y_train = pd.read_csv(f"/data/y_train.csv")

X_val = np.asarray(pd.read_csv(f"/data/X_val.csv"), dtype=np.float32)
X_val = X_val.reshape(-1, 28, 28, 1)
y_val = pd.read_csv(f"/data/y_val.csv")

train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)

train_part(model_init_fn, optimizer_init_fn, num_epochs=num_epochs, is_training=True)