import numpy as np
import os
import csv
import cv2
import tensorflow as tf

def parse(serialized):

    features={
        'label':tf.FixedLenFeature([],tf.int64),
        'image':tf.FixedLenFeature([],tf.string)
    }
    
    parsed_example = tf.parse_single_example(serialized = serialized,features = features)
    image = parsed_example['image']
    image = tf.image.decode_image(image)
    image = tf.cast(image,tf.float32)
    image = image
    print(image.shape)
    image = tf.reshape(image,[28,28,3])
    label = parsed_example['label']
    label = tf.cast(label,tf.int32)
    return image,label

def input_func(filename,batch_size = 10):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse,16)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(20)
    dataset = dataset.prefetch(batch_size)
    return dataset