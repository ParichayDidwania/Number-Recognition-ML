import csv
import os
import cv2
import tensorflow as tf

dir = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/processed_images_2'
csv_dir = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/label.csv'

images_windows = []
for i in os.listdir(dir):
    images_windows.append(i)

images_csv = []
labels_csv = []
with open(csv_dir) as csv_file:
    read = csv.reader(csv_file)
    for i in read:
        images_csv.append(i[0])
        labels_csv.append(i[1])

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tfrecord_filename = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecord_filename)
count = 0

for i in images_windows:
    count+=1
    index = images_csv.index(i.split('.')[0])
    with open(dir+'/'+i,'rb') as img_file:
      image_feature = img_file.read()
    
    label_feature = int(labels_csv[index])

    feature = {
                'label': _int64_feature(label_feature),
                'image': _bytes_feature(image_feature),
                }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()
print(count)

