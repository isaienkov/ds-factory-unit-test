import pandas as pd

import tensorflow as tf


def preprocess_pandas_df():
    df = pd.DataFrame(columns=['A', 'B', 'C', 'D'])
    df = df.drop(['C'], axis=1)
    return df


def get_mnist_image():
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:1]
    return x_train


def preprocess_image(img):
    img = img.astype("float32") / 255
    return img
