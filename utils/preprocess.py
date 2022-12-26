#Data sorter
import re
import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 128

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


#Define the data loader
def data_loader(path, split=0.05):
    images = glob(os.path.join(path, "images/*"))
    images.sort(key=natural_keys)
    
    masks = glob(os.path.join(path, "masks/*"))
    masks.sort(key=natural_keys)
    
    total_size = len(images)
    valid_size = int(split * total_size)
    
    train_x, val_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, val_y = train_test_split(masks, test_size=valid_size, random_state=42)
     
    return (train_x, train_y), (val_x, val_y)


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    y.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

def read_and_rgb(x):
    x = cv2.imread(x)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x