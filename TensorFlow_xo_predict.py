# -*- coding: utf-8 -*-
"""
Created on Wed Apr 4 2017  
based on tutorials for Tensorflow
https://www.tensorflow.org/get_started/
@author: artur
"""
import tensorflow as tf
import numpy
import time
import TensorFlow_XO_dataReadIn
import os
from tensorflow.python.framework import graph_util

"""
-load data from  images
#format [number of image, heigh, width,depth=1]
1000 images from each category
categorylabels  - one-hot array
    0.       -circle
    1.       -cross
    
    
    e.g. [0,1] -> a cross
""" 
  
pathtoFile='data/' #path to training_data_sm folder

pathtoFile = os.getcwd() + "/"

summary_path = os.getcwd() + "/summary/"

print(pathtoFile)

[images_train, image_labels_train, images_test, image_labels_test, images_valid,
     image_labels_valid] = TensorFlow_XO_dataReadIn.data_readIn_and_subdivision_XoXo(pathtoFile) #reads the image data from files and randomly divides into train,valid and test

     
print("Curr: ", os.getcwd())
#function to make a string with hyperparameters to add to summaryname, easy differentiation of runs in Tensorboard
def make_hyper_string(Nodes_fullyconnected, Kernelsize, maxPooling, ActivationFunc):
        hyperstring='FCneurons=' + str(Nodes_fullyconnected) + '_Kernelsize=' + str(Kernelsize) + '_maxPool=' +str(maxPooling) + '_ActFunc=' + str(ActivationFunc)
        return hyperstring
        
# ==============================================================================
# Save .pb and log its .pbtxt with
# (1) Remove all the explicit device specifications
# (2) Convert variables to constants
# (3) Remove training nodes
# ==============================================================================
def Save_pb(sess, net, pbName, clear_devices = True):
    output_node_names = net.name
    output_node_names =output_node_names.split(":")
    output_node_names = output_node_names[0]
    
    input_graph_def = net.graph.as_graph_def()
    
    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ""
    
    output_graph_def = graph_util.convert_variables_to_constants( 
        sess,
        input_graph_def,
        output_node_names.split(",")
    )
    
    output_graph_def = graph_util.remove_training_nodes(output_graph_def)
    tf.train.write_graph(output_graph_def, './',pbName+"_MobileDeploy.pbtxt")
    print("pbtxt file generated ^____^")
    
    with tf.gfile.GFile(pbName+"_frozen.pb", "wb") as f: 
        f.write(output_graph_def.SerializeToString())
    print("pb file generated ^____^")
    print("%d ops in the final graph." % len(output_graph_def.node))

activationList = {0: tf.nn.tanh, 1: tf.sigmoid, 2: tf.nn.relu, 3: tf.nn.elu} #dictionary for choosing the activation function

IMAGESIZE = images_train.shape[2]
IMAGEDEPTH = images_train.shape[3]
NUMBER_categories = image_labels_train.shape[1]
Filters_layer1 = 16
Filters_layer2 = 16
Filters_layer3 = 16
Nodes_fullyconnected = 20
BATCH_size = 100
TRAIN_STEPS = 200
Kernelsize=5 #size of the kernel ex. 5x5
maxPoolingSize=2 #
ActivationFunc=2

#with tf.device('/cpu:0'): # uncomment to run on the CPU
tf.reset_default_graph()  # reset the default graph when several runs are executed inside a for loop 
      
#define placeholders
#image input is a tensor [batchFile,ImageWidth,ImageHeight,Imagedepth]
Input = tf.placeholder(tf.float32, shape = [None, IMAGESIZE,IMAGESIZE,IMAGEDEPTH], name='Image_Input') #None- placeholder for different batchsizes
Output_label = tf.placeholder(tf.float32, shape = [None, NUMBER_categories], name='Labels_Input')           
  
tf.summary.image("ExampleInput", Input,4) #add 4 example pictures to summary
 
#First convolutional layer
with tf.name_scope('Conv1'):
    conv1=tf.layers.conv2d(Input,Filters_layer1,Kernelsize,activation=activationList[ActivationFunc],name='Conv1')#convolution
    maxp1=tf.layers.max_pooling2d(conv1, maxPoolingSize, 2, padding='valid', name='maxPool1') #max pooling
    with tf.variable_scope('Conv1', reuse=True): #add  kernels pictures to summary
      weights=tf.get_variable('kernel') 
      tf.summary.image("ExampleKernels", tf.transpose(weights,[3,0,1,2]),Filters_layer1) #adds kernels to the summary as images
      tf.summary.histogram("weightsConv1",weights) #adds the distribution of kernels to the summary                                
    
#Second convolutional layer      
with tf.name_scope('Conv2'):
    conv2=tf.layers.conv2d(maxp1,Filters_layer2,Kernelsize,activation=activationList[ActivationFunc],name='Conv2')
    maxp2=tf.layers.max_pooling2d(conv2, maxPoolingSize, 2, padding='valid', name='maxPool2')       

#Third convolutional layer
with tf.name_scope('Conv3'):
    conv3=tf.layers.conv2d(maxp2,Filters_layer3,Kernelsize,activation=activationList[ActivationFunc],name='Conv3')
    maxp3=tf.layers.max_pooling2d(conv3, maxPoolingSize, 2, padding='valid', name='maxPool3')             

#Fully connected layers
print("maxp3.shape: ", maxp3.shape)
size1D = maxp3.shape[1] * maxp3.shape[2] * maxp3.shape[3]
print("size1D: ", size1D)
h_pool3_flat = tf.reshape(maxp3, [-1, size1D]) #reshape to input a vector to fully connected layer
# h_pool3_flat = tf.contrib.layers.flatten(maxp3)#reshape to input a vector to fully connected layer
FClayer1 = tf.layers.dense(inputs=h_pool3_flat, units=Nodes_fullyconnected, activation=activationList[ActivationFunc], name='FC1')
Outputlayer = tf.layers.dense(inputs=FClayer1, units=NUMBER_categories, activation=None,name='OutputLayer') #output layer
    
#Evaluate and train
#cross entropy of output as logits and labels
with tf.name_scope('CrossEntropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Output_label, logits=Outputlayer))#calculate loss as cross entropy
    tf.summary.scalar("cost", cross_entropy)

#train using Adamoptimizer    
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  
with tf.name_scope('AccuracyEval'):
    correct_prediction = tf.equal(tf.argmax(Outputlayer,1), tf.argmax(Output_label,1))# check if prediction was right
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# calculate accuracy
    tf.summary.scalar("accuracy1", accuracy)
  

summary_op = tf.summary.merge_all()# merge all summaries for one-line evaluation
  
#define where summary is to be written
writer = tf.summary.FileWriter(summary_path + make_hyper_string(Nodes_fullyconnected, Kernelsize, maxPoolingSize, ActivationFunc)+ '--gpu', graph=tf.get_default_graph() )

#initialize variables
sess = tf.InteractiveSession() # a session for the graph execution
sess.run(tf.global_variables_initializer())

#indent until here to use CPU 
   
reached95 = 0
reached99 = 0      
epoch=-1
t0=time.time()

for i in range(TRAIN_STEPS): #Train for 2000 steps
    if i%(images_train.shape[0]/BATCH_size) == 0: #shuffle data after all was used 
        [images_train, image_labels_train]= TensorFlow_XO_dataReadIn.shuffle_data(images_train, image_labels_train)
        ii=0
        epoch=epoch+1
        print("Epoch: %d"%(epoch)) 
        
    # get a batch of data for training
    batch_im = images_train[BATCH_size*ii:(BATCH_size-1)+ii*BATCH_size,:,:,:]
    batch_lab = image_labels_train[BATCH_size*ii:(BATCH_size-1)+ii*BATCH_size,:]  
    #print out training accuracy and write summary every 10 steps
    if i%10 == 0:
      train_accuracy = accuracy.eval(feed_dict={Input:batch_im, Output_label: batch_lab})
      print("step %d, training accuracy %g"%(i, train_accuracy))   #output to console
      summary=sess.run(summary_op, feed_dict={Input: batch_im, Output_label: batch_lab}) #execute summary op
      writer.add_summary(summary,i)     #write summary timepoint to disk
      
    if train_accuracy > 0.95 and reached95 == 0: # if 95 % training accuracy reached test validation set and print time required
          reached95 = numpy.round(time.time()-t0)        
          print('Time passed to reach 95%:', reached95, ' seconds')
          print("ValidationSet accuracy %g"%accuracy.eval(feed_dict={
              Input: images_valid, Output_label: image_labels_valid}))    
    if train_accuracy > 0.99 and reached99 == 0:
          reached99 = numpy.round(time.time()-t0)    
          print('Timepassed To reach 99%:', reached99, ' seconds')
          print("ValidationSet accuracy %g"%accuracy.eval(feed_dict={Input: images_valid, Output_label: image_labels_valid}))
    train_step.run(feed_dict={Input: batch_im, Output_label: batch_lab})#actual execution of the training-op 
    ii=ii+1 # increase step        

images_after_activation1=sess.run(maxp1,feed_dict={Input:images_train[50:52,:,:,:]}) #get few examples of image  after convolution
images_after_activation2=sess.run(maxp2,feed_dict={Input:images_train[50:52,:,:,:]})
images_after_activation3=sess.run(maxp3,feed_dict={Input:images_train[50:52,:,:,:]})
print('Time passed To reach 95%:', reached95, ' seconds')
print('Time passed To reach 99%:', reached99, ' seconds')
print('Time passed To end:', numpy.round(time.time()-t0), ' seconds')
print("ValidationSet accuracy %g"%accuracy.eval(feed_dict={
          Input: images_valid, Output_label: image_labels_valid}))

Save_pb(sess, Outputlayer, "OX_Predict")       
sess.close() #close the session
