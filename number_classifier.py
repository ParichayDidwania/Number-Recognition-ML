import tensorflow as tf
import tqdm
import numpy as np
from reading_tfrecord import input_func
import matplotlib.pyplot as plt

def model(in_data,keep_prob):
    
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[4, 4, 3, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[4, 4, 64, 128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[4, 4, 128, 256], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], mean=0, stddev=0.08))
    x=tf.layers.batch_normalization(in_data)
    
    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)
    
    print(conv1_bn.shape)

    '''
    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')    
    conv2_bn = tf.layers.batch_normalization(conv2_pool)
    
   
  
    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
    conv3_bn = tf.layers.batch_normalization(conv3_pool)
   
    
    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)
    print(conv4_bn.shape)
    '''
    # 9
    flat = tf.contrib.layers.flatten(conv1_bn)  
    
    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.layers.batch_normalization(full1)
    full1 = tf.nn.dropout(full1, keep_prob)
    
    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.layers.batch_normalization(full2)
    full2 = tf.nn.dropout(full2, keep_prob)
    
    # 12
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.layers.batch_normalization(full3)  
    full3 = tf.nn.dropout(full3, keep_prob)  
    
    # 13
    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=512, activation_fn=tf.nn.relu)
    full4 = tf.layers.batch_normalization(full4)
    full4 = tf.nn.dropout(full4, keep_prob)    
    
    # 14
    out = tf.contrib.layers.fully_connected(inputs=full4, num_outputs=10, activation_fn=None)
    return out

def train_model(data,epoch=30):
    x_place = tf.placeholder(tf.float32,[None,28,28,3],name='x_place')
    y_place = tf.placeholder(tf.int32,[None],name='y_place')
    infer_data = tf.data.Dataset.from_tensor_slices((x_place,y_place))
    infer_data = infer_data.batch(100)
    Iterator = tf.data.Iterator.from_structure(data.output_types,data.output_shapes)
    next_image,next_label = Iterator.get_next()
    Y = tf.one_hot(next_label,10)
    Y = tf.cast(Y,tf.int32)
    logits = model(next_image,0.9)

    
    
    train_op = Iterator.make_initializer(data,name='train_op')
    test_op = Iterator.make_initializer(infer_data,name='test_op')

    
    with tf.name_scope('loss'):
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y),name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

        prediction = tf.argmax(logits,1,name='pred')
        equal = tf.equal(prediction,tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))

        tf.summary.scalar('loss',loss)
        tf.summary.scalar('accuracy',accuracy)

        merge = tf.summary.merge_all()

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

    j=0
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/trained_graph',sess.graph)
        for i in range(epoch):
            sess.run(train_op)
            if(i>=2):
                saver.save(sess,"C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/trained_graph/number_classifier"+str(j))
            while(True):
                try:
                    j+=1
                    if j%100!=0:
                        summ,_ = sess.run([merge,optimizer])
                        writer.add_summary(summ,j)
                    else:
                        
                        l,_,acc = sess.run([loss,optimizer,accuracy])
                        if(j==1 or j%20==0):
                            print("iters: {}, loss: {:.10f}, training accuracy: {:.2f}".format(j, l, acc*100))
                    
                except tf.errors.OutOfRangeError:
                    break
            
        saver.save(sess,"C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/trained_graph/number_classifier"+str(j))
            
            
            


            
data = input_func('C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/train.tfrecords')
train_model(data)


    
