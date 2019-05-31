import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

path = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/sample/4_digit.jpg'
dir_sample = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/sample'

def read_convert(path):
    img = cv2.imread(path,0)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #img = (cv2.bitwise_not(img))   #Uncomment if image contains digits in BLACK
    h,w,c = img.shape
    for i in range(h):
        for j in range(w):
            if(img[i,j,0]>127 and img[i,j,1]>127 and img[i,j,2]>127):
                img[i,j,0]=255
                img[i,j,1]=255
                img[i,j,2]=255
            else:
                img[i,j,0]=0
                img[i,j,1]=0
                img[i,j,2]=0
    cv2.imshow('f',img)
    cv2.waitKey(0)
    return img



def image_vertical_strip(img):
    f=0
    row_val=[]
    count=0
    h,w,c = img.shape
    print(h,w)
    for i in range(h):
        for j in range(w):              
            if(img[i,j,0]==255 and f==0):
                row_val.append(i)
                f=1
                break
    for i in range(h-1,0,-1):
        for j in range(w-1,0,-1):
            if(img[i,j,0]==255 and f==1):
                row_val.append(i)
                f=2
                break
    img = img[row_val[0]:row_val[1],0:w]
    cv2.imshow('f',img)
    cv2.waitKey(0)
    print(row_val)
    return img

def seperation(img):
    f=0
    count = 0
    single_coord=[]
    col_val=[]
    h,w,c = img.shape
    start=0
    end=0
    for j in range(w):
        for i in range(h):
            if(img[i,j,0]==0 and f==1):
                count+=1
            if(img[i,j,0]==255 and f==0):
                start = j
                f=1
                break            
        if(count==h and f==1):
            end = j
            single_coord.append(start)
            single_coord.append(end)
            col_val.append(single_coord)
            single_coord=[]
            f=0
        count=0
    return col_val,img

def image_set(l,img):
    h,w,c = img.shape
    images = []
    for i in l:
        image = img[0:h,i[0]:i[1]]
        image = cv2.copyMakeBorder( image, 5, 5, 5, 5, cv2.BORDER_CONSTANT,value=[0,0,0])
        image = cv2.resize(image,(28,28))
        images.append(image)
    
    return images


img = read_convert(path)
img = image_vertical_strip(img)
l,img = seperation(img)
images = image_set(l,img)

for i in images:
    cv2.imshow('f',i)
    cv2.waitKey(0)

x = np.array(images)           

garb=np.random.rand(x.shape[0])
print(garb.shape)

with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/trained_graph/number_classifier4201.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint('C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/trained_graph/'))
    graph = tf.get_default_graph()
    dataset_init_op = graph.get_operation_by_name('test_op')
    x_input=graph.get_tensor_by_name("x_place:0")
    y=graph.get_tensor_by_name("y_place:0")
    logit_2 = graph.get_tensor_by_name("fully_connected_4/BiasAdd:0")
    logit=graph.get_tensor_by_name("loss/pred:0")
    sess.run(dataset_init_op,feed_dict={x_input:x,y:garb})
    
    log=sess.run(logit)
    print(log)
