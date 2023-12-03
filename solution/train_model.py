from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import tensorflow
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical


wine = load_wine(as_frame=True)
wine.frame.head()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=123)

train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)
def declare_model_for_learning():
    model = models.Sequential()
    model.add(layers.Dense(12, activation='relu', input_shape=(13,)))
    model.add(layers.Dense(12, activation='relu')) 
    model.add(layers.Dense(3, activation='softmax'))
    return model

my_model = declare_model_for_learning()
my_model.compile(optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
my_model.save_weights('/data/model.h5')

my_model.load_weights('/data/model.h5')
csv_logger = [tensorflow.keras.callbacks.CSVLogger('/data/log.csv', append=True, separator=';')]
model_history = my_model.fit(X_train, train_labels, epochs=100, batch_size=1, verbose=2, callbacks=csv_logger)