import pandas as pd
import tensorflow as tf


if __name__ == "__main__":
    boston = tf.keras.datasets.boston_housing
    (train_data, train_targets), (test_data, test_targets) = boston.load_data()

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    df_train = pd.DataFrame(data=train_data, columns=[f'feature_{i}' for i in range(train_data.shape[1])])
    df_train['target'] = train_targets

    df_train.to_csv('/data/boston_train_data.csv', index=False)
