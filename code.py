import numpy as np
import os,math,cv2, h5py
import tensorflow as tf
from collections import defaultdict
from scipy import misc
from scipy import spatial

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
from random import shuffle
from sklearn.cluster import MiniBatchKMeans
import glob

alex_net_path = os.path.join("tf_models/bvlc_alexnet.npy")
alex_net = np.load(alex_net_path, encoding='latin1').item()

vgg_net_path = os.path.join("tf_models/vgg16.npy")
vgg_net = np.load(vgg_net_path, encoding='latin1').item()
print(alex_net.keys())
print(vgg_net.keys())

a_c1 = alex_net['conv1']
w1 = a_c1[0]
b1 = a_c1[1]
print(w1.shape)

#Needed for creating feature descriptors
def max_pool(input_x, kernel_size, stride, padding='VALID'):
  ksize = [1, kernel_size, kernel_size, 1]
  strides = [1, stride, stride, 1]
  return tf.nn.max_pool(input_x, ksize=ksize, strides=strides, padding=padding)

#Here we already have pre-trained weights
def conv_2d(input_x, weights, stride, bias=None, padding='VALID'):
  stride_shape = [1, stride, stride, 1]
  c = tf.nn.conv2d(input_x, weights, stride_shape, padding=padding)
  if bias is not None:
    c += bias
  return c


def imgread(path):
  print("Image:", path.split("/")[-1])
  # Read in the image using python opencv
  img = cv2.imread(path)
  img = img / 255.0
  print("Raw Image Shape: ", img.shape)
  
  # Center crop the image
  short_edge = min(img.shape[:2])
  W, H, C = img.shape
  to_crop = min(W, H)
  cent_w = int((img.shape[1] - short_edge) / 2)
  cent_h = int((img.shape[0] - short_edge) / 2)
  img_cropped = img[cent_h:cent_h+to_crop, cent_w:cent_w+to_crop]
  print("Cropped Image Shape: ", img_cropped.shape)
  
  # Resize the cropped image to 224 by 224 for VGG16 network
  img_resized = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
  print("Resized Image Shape: ", img_resized.shape)
  return img_resized


ip = tf.Variable(tf.random_normal([1,3,3,5]))
ft = tf.Variable(tf.random_normal([1,1,5,1]))

def normalize(ip):
  m2 = np.min(ip)
  ip = ip - m2
  m1 = np.max(ip)
  ip = ip / m1
  return ip

def alex_net_graph(ip, weights, biases):
  w1, w2, w3, w4, w5 = weights
  b1, b2, b3, b4, b5 = biases
  with tf.variable_scope("alex_net"):
    #CONV 1
    c1 = conv_2d(ip, w1, 4, b1, padding='VALID')
    r1 = tf.nn.relu(c1)
    m1 = max_pool(r1, 3, 2, padding='VALID')
    #print("M1", m1.get_shape)
    
    #CONV2
    m1 = tf.pad(m1, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT") # add 2 padding
    i1, i2 = tf.split(axis = 3, num_or_size_splits=2, value=m1)
    w2_1, w2_2 = tf.split(axis = 3, num_or_size_splits=2, value=w2)
    o1 = conv_2d(i1, w2_1, 1, bias=None, padding='SAME')
    o2 = conv_2d(i2, w2_2, 1, bias=None, padding='SAME')
    c2 = tf.concat(axis = 3, values = [o1,o2])
    r2 = tf.nn.relu(c2)
    m2 = max_pool(r2, 3, 2, padding='VALID')
    #print("M2",m2.get_shape)
    
    #CONV3
    c3 = conv_2d(m2, w3, 1, b3)
    r3 = tf.nn.relu(c3)
    #print(r3.get_shape, "R3")
    
    #CONV4
    i1, i2 = tf.split(axis = 3, num_or_size_splits=2, value=r3)
    w4_1, w4_2 = tf.split(axis = 3, num_or_size_splits=2, value=w4)
    o1 = conv_2d(i1, w4_1, 1, bias=None, padding='SAME')
    o2 = conv_2d(i2, w4_2, 1, bias=None, padding='SAME')
    c4 = tf.concat(axis = 3, values = [o1,o2])
    r4 = tf.nn.relu(c4)
    #print(r4.get_shape, "R4")
    
    #CONV5
    i1, i2 = tf.split(axis = 3, num_or_size_splits=2, value=r4)
    w5_1, w5_2 = tf.split(axis = 3, num_or_size_splits=2, value=w5)
    o1 = conv_2d(i1, w5_1, 1, bias=None, padding='SAME')
    o2 = conv_2d(i2, w5_2, 1, bias=None, padding='SAME')
    c5 = tf.concat(axis = 3, values = [o1,o2])
    r5 = tf.nn.relu(c5)
    m5 = max_pool(r5, 3, 2, padding='VALID')
    #print(m5.get_shape, "M5")
    
    layers = [m1,m2,r3,r4,m5]
    return layers

def features_alex_net(inputs, alex_net):
  tf.reset_default_graph()
  H,W,D = 227, 227, 3
  
  w1, b1 = alex_net['conv1'][0], alex_net['conv1'][1]
  w2, b2 = alex_net['conv2'][0], alex_net['conv2'][1]
  w3, b3 = alex_net['conv3'][0], alex_net['conv3'][1]
  w4, b4 = alex_net['conv4'][0], alex_net['conv4'][1]
  w5, b5 = alex_net['conv5'][0], alex_net['conv5'][1]
  
  weights = [w1,w2,w3,w4,w5]
  biases = [b1,b2,b3,b4,b5]
  
  #print(w1.shape, w2.shape, w3.shape, w4.shape, w5.shape)
  
  images = tf.placeholder(tf.float32, [None, H, W, D])
  input_layers = alex_net_graph(images, weights, biases)
  init = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init)
    result  = sess.run(input_layers, feed_dict={images: inputs})
    return result

H,W,D = 227, 227, 3
training_list = []
training_list_names = []

for filename in sorted(glob.glob('training/*.jpg')):
  img = cv2.imread(os.path.join(filename))
  img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
  training_list.append(img)
  training_list_names.append(filename)

def run(images,alex_net):
  conv1, conv2, conv3, conv4, conv5 = features_alex_net(images, alex_net)

  m1 = np.amax(conv1, axis=(1,2))
  m2 = np.amax(conv2, axis=(1,2))
  m3 = np.amax(conv3, axis=(1,2))
  m4 = np.amax(conv4, axis=(1,2))
  m5 = np.amax(conv5, axis=(1,2))

  return np.concatenate((m1,m2,m3,m4,m5), axis=1)

r = run(training_list, alex_net)

for filename in sorted(glob.glob('testing/*.jpg')):
  img = cv2.imread(os.path.join(filename))
  img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
  solution = run([img], alex_net)
  closestIdx = spatial.cKDTree(r).query(solution[0], k=1)[1]
  print(closestIdx)
  print(training_list_names[closestIdx])

# solution = run([i1], alex_net)
# closestIdx = spatial.cKDTree(r).query(solution[0], k=1)[1]

# print(closestIdx)
