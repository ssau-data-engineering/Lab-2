import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score



data_train = pd.read_csv('/data/mnist_train.csv')
data_test = pd.read_csv('/data/mnist_test.csv')

print(f'Dataset MNIST - объёмная база данных образцов рукопасного написания цифр.\nРазмерность train: {data_train.shape}, test: {data_test.shape}')

# Уменьшение размерности в условиях ограниченых ресурсов
data_train = data_train[:1000]
data_test = data_test[:100]

print(f'Новая размерность: train: {data_train.shape}, test: {data_test.shape}')

# Подготовка значений признаков
data_train_not_label = data_train[:]
data_test_not_label = data_test[:]

del data_train_not_label['label']
del data_test_not_label['label']

X_train = data_train_not_label
X_test = data_test_not_label

# Нормировка данных
X_train /= 255
X_test /= 255

# Преобразование в numpy массив
X_train = data_train_not_label.to_numpy()
X_test = data_test_not_label.to_numpy()
print(f'Размерность признаков, train: {X_train.shape}, test: {X_test.shape}.')

# Подготовка меток классов при помощи OneHot алгоритма для кодировки
from sklearn.preprocessing import OneHotEncoder

encoding = OneHotEncoder(sparse_output=False, handle_unknown='error')
labels = pd.concat([data_train[['label']], data_test[['label']]])
encoding.fit(labels)

y_train = pd.DataFrame(encoding.transform(data_train[['label']]))
y_test = pd.DataFrame(encoding.fit_transform(data_test[['label']]))

# Преобразование в numpy массив
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
print(f'Размерность меток классов, train:  {y_train.shape}, test: {y_test.shape}.\n\n {y_train[0:6]}')

# Функция активакии Relu
def relu(x):
    return (x > 0) * x

# Производная функции активации
def relu_two_deriv(x):
    return x>0

# Модель предсказания трехслойной системы
def model_prediction(input, weights_0_1, weights_1_2):
    layer_0 = input
    layer_1 = relu(np.dot(layer_0,weights_0_1))
    layer_2 = np.dot(layer_1,weights_1_2)
    return layer_2


# Гиперпараметры сети
np.random.seed(1)
learning_rate = 0.005
hidden_size = 100
input_size = 784
num_labels = 10

# Веса между входным и скрытым слоями
weights_input_hidden = (2*np.random.random((input_size,hidden_size)) - 1) / np.sqrt(input_size)
# Веса между скрытым и выходным слоями
weights_hidden_output = (2*np.random.random((hidden_size, num_labels)) - 1) / np.sqrt(hidden_size)

# Массивы для записи
array_iteration = []
array_error = []
array_accuracy = []

text_file = open("/data/train.txt", "w+")

for iteration in range(10):
    error = 0
    for i in range(len(X_train)):
     # Прямой проход
        input_layer = X_train[i:i+1]
        hidden_layer = relu(np.dot(input_layer,weights_input_hidden))
        output_layer = np.dot(hidden_layer,weights_hidden_output)

        error = error + np.sum((output_layer - y_train[i:i+1]) ** 2)

        # Обратный проход
        diff_between_output_hidden = 2*(output_layer - y_train[i:i+1])
        weight_hidden_output_corrections = hidden_layer.T.dot(diff_between_output_hidden) # Величина коррекции для весов между layer_1 и layer_2

        diff_between_input_hidden = diff_between_output_hidden.dot(weights_hidden_output.T)*relu_two_deriv(hidden_layer)
        weight_input_hidden_corrections = input_layer.T.dot(diff_between_input_hidden)

        weights_hidden_output = weights_hidden_output - learning_rate * weight_hidden_output_corrections
        weights_input_hidden = weights_input_hidden - learning_rate * weight_input_hidden_corrections

    if(iteration % 2 == 0):
        print(f'Iteration = {iteration}\nError: {error}')
        array_iteration.append(iteration)
        array_error.append(error)
        goal_idx = np.argmax(y_train[:100], axis=1)
        y_prediction = model_prediction(X_train[:100], weights_input_hidden, weights_hidden_output)
        pred_idx = np.argmax(y_prediction, axis=1)
        accuracy = accuracy_score(goal_idx ,pred_idx)
        array_accuracy.append(accuracy)
        print(f'Accuracy: {accuracy}\n')

        text_file.write(str(' |Iteration '))
        text_file.write(str(iteration))

        text_file.write(str(' |Erorr '))
        text_file.write(str(error))

        text_file.write(str(' |Accuracy '))
        text_file.write(str(accuracy))
        text_file.write(str('\n'))
text_file.close()