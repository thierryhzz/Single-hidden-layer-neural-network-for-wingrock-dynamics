## Demo Code for Tensor Flow
## Author: Zhong Zhang
## Advisor/Prof: Girish Chowdhary
## Created as in class demo for ABE 598, Spring 2017
## Date: March 16 2017
## Copyright: All copyright held by UIUC and DASLAB, contact: girishc@illinois.edu

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


import matplotlib.pyplot as plt # this is needed for plotting later
import numpy as np
import tensorflow as tf

W = tf.Variable(tf.zeros([2, 10]), tf.float32) # 784 is the size of the input space, each image is 28 by 28, and its flattened to a single row vector. 500 is the hidden layer neurons
b = tf.Variable(tf.zeros([10]), tf.float32) # bias for the hidden layer neuron
x = tf.placeholder(tf.float32, [None, 2]) # place holder

activation = tf.matmul(x, W) + b
#activation = tf.nn.relu(activation) # we just relu'd the output of the hidden layer
activation = tf.nn.sigmoid(activation)
# now we define the outer layer
W_out = tf.Variable(tf.zeros([10,1]), tf.float32) # 784 is the size of the input space, each image is 28 by 28, and its flattened to a single row vector. 500 is the hidden layer neurons
b_out = tf.Variable(tf.zeros([1]), tf.float32) # bias for the hidden layer neuron
y = tf.matmul(activation, W_out) + b_out



y_ = tf.placeholder(tf.float32, [None, 1])

loss = tf.reduce_mean(tf.square(y_-y))

train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#train = tf.train.AdamOptimizer(0.005).minimize(loss)



# variables for plotting later
loss_list = []
ilist = []

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

filename_queue = tf.train.string_input_producer(["X_IN_Y_TARGET.txt"])
reader=tf.TextLineReader()
key,value=reader.read(filename_queue)
record_defaults=[[1.000000],[1.000000],[1.000000]]
col1,col2, col3=tf.decode_csv(value, record_defaults=record_defaults)
xin=tf.stack([col1,col2])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)



example= sess.run([xin,col3])
#print(example)
batch_x=np.array([example[0]], np.float32)
batch_y=np.array([example[1]], np.float32)

for i in range(3995):
    # Retrieve a single instance:
  example= sess.run([xin,col3])
  #print(example)
  batch_x = np.vstack((batch_x, [example[0]]))

  batch_y = np.vstack((batch_y, [example[1]]))

for i in range(10000):

  _, loss_val = sess.run([train, loss], {x:batch_x, y_:batch_y})

        #end for
  if i % 100 == 0:
    print('loss =', loss_val)

prediction=(sess.run(y,{x:batch_x}))

plt.figure(1)
plt.plot(range(3996),prediction)
#plt.ylabel('Loss')
#plt.xlabel('iteration')
plt.show()

plt.figure(2)
plt.plot(range(3996),batch_y)
#plt.ylabel('Loss')
#plt.xlabel('iteration')
plt.show()
