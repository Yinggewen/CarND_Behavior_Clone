
# coding: utf-8

# 
# Use Keras to train a network to do the following:
# 
# 1. Take in an image from the center camera of the car. This is the input to your neural network.
# Output a new steering angle for the car.
# 
# 2. You don’t have to worry about the throttle for this project, that will be set for you.
# 
# 3. Save your model architecture as model.json, and the weights as model.h5.
# https://keras.io/models/about-keras-models/
# 
# 4. python drive.py model.json
# 
# 5. Tips
# Adding dropout layers to your network.
# Splitting your dataset into a training set and a validation set.
# 
# 

# In[1]:

##get_ipython().magic('matplotlib inline')

import argparse
import numpy as np
import tensorflow as tf
import os
from os import listdir
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread
from scipy import ndimage
from skimage.exposure import adjust_gamma,rescale_intensity
from sklearn.model_selection import train_test_split
import pandas as pd


# ## Step 0: Load The Data

# In[2]:

df = pd.read_csv("driving_log_clean_all.csv", header=None)
df.columns=('Center Image','Left Image','Right Image','Steering Angle')
print (df.shape)
df.head()


# In[3]:

listdir=os.listdir("/home/ubuntu/udacity_wu/CarND-behavior-clone/test2/final/IMG/")
listdir.sort()


# In[4]:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


'''

# In[5]:

center_data=df['Center Image']
left_data=df['Left Image']
right_data=df['Right Image']


# In[6]:

def load_images_from_center(folder):
    images = []
    for filename in center_data:
        #img = cv2.imread(os.path.join(folder,filename))
        #img = mpimg.imread(filename.strip())
        img = mpimg.imread(filename.strip())
        if img is not None:
            images.append(img)
    images=np.stack(images, axis=0)
    return images


def load_images_from_left(folder):
    images = []
    for filename in left_data:
        #img = cv2.imread(os.path.join(folder,filename))
        img = mpimg.imread(filename.strip())
        #print(filename)
        #print(len(img.shape))
        #print(filename)
        #break
        if img is not None:
            images.append(img)
    images=np.stack(images, axis=0)
    return images

def load_images_from_right(folder):
    images = []
    for filename in right_data:
        #img = cv2.imread(os.path.join(folder,filename))
        img = mpimg.imread(filename.strip())
        if img is not None:
            images.append(img)
    images=np.stack(images, axis=0)
    return images


# In[7]:

image_center=load_images_from_center("/home/ubuntu/udacity_wu/CarND-behavior-clone/test2/IMG/")
image_left=load_images_from_left("/home/ubuntu/udacity_wu/CarND-behavior-clone/test2/IMG/")
image_right=load_images_from_right("/home/ubuntu/udacity_wu/CarND-behavior-clone/test2/IMG/")


# In[8]:

plt.imshow(image_center[0])


# In[9]:

plt.imshow(image_right[0])


# In[10]:

plt.imshow(image_left[0])

'''


# ## Step 1:  Preprocess the data 

# ### Populate controlled driving datasets

# In[3]:

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('imgs_dir', 'IMG/', 'The directory of the image data.')
flags.DEFINE_string('data_path', 'driving_log_clean_all.csv', 'The path to the csv of training data.')
flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
flags.DEFINE_integer('num_epochs', 5, 'The number of epochs to train for.')
flags.DEFINE_float('lrate', 0.0001, 'The learning rate for training.')


# In[4]:


def read_imgs(img_paths):
    imgs = np.empty([len(img_paths), 160, 320, 3])

    for i, path in enumerate(img_paths):
        imgs[i] = imread(path)

    return imgs

def resize(imgs, shape=(32, 16, 3)):
    """
    Resize images to shape.
    """
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = imresize(img, shape)

    return imgs_resized

def rgb2gray(imgs):
    """
    Convert images to grayscale.
    """
    return np.mean(imgs, axis=3, keepdims=True)

def normalize(imgs):
    """
    Normalize images between [-1, 1].
    """
    return imgs / (255.0 / 2) - 1

def preprocess(imgs):
    imgs_processed = resize(imgs)
    imgs_processed = rgb2gray(imgs_processed)
    imgs_processed = normalize(imgs_processed)

    return imgs_processed

#Augment the data by randomly flipping some angles / images horizontally.
def random_flip(imgs, angles):    
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        if np.random.choice(2):
            new_imgs[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_imgs[i] = img
            new_angles[i] = angle

    return new_imgs, new_angles

def augment(imgs, angles):
    imgs_augmented, angles_augmented = random_flip(imgs, angles)

    return imgs_augmented, angles_augmented


# In[5]:

def gen_batches(imgs, angles, batch_size):
    """
    Generates random batches of the input data.

    :param imgs: The input images.
    :param angles: The steering angles associated with each image.
    :param batch_size: The size of each minibatch.

    :yield: A tuple (images, angles), where both images and angles have batch_size elements.
    """
    num_elts = len(imgs)

    while True:
        indeces = np.random.choice(num_elts, batch_size)
        batch_imgs_raw, angles_raw = read_imgs(imgs[indeces]), angles[indeces].astype(float)
        batch_imgs, batch_angles = augment(preprocess(batch_imgs_raw), angles_raw)

        yield batch_imgs, batch_angles


# ## Step 2:  Split the data 

# In[6]:

# Construct arrays for center, right and left images of controlled driving
angles = np.array(df['Steering Angle'])
center = np.array(df['Center Image'].map(str.strip))
right = np.array(df['Right Image'].map(str.strip))
left = np.array(df['Left Image'].map(str.strip))


# In[7]:

# Concatenate all arrays in to combined training dataset and labels
X_train = np.concatenate((center, right, left), axis=0)
y_train = np.concatenate((angles, angles-np.array(0.08), angles+np.array(0.08)),axis=0)

# Perform train/test split to a create validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.05)


# ## Step 3: Design and Test a Model Architecture

# In[8]:

import json
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[9]:

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),)
model.add(Conv2D(24, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
model.add(Dropout(.5))
model.add(Conv2D(36, 3, 3, border_mode='valid', activation='relu'))
model.add(Conv2D(48, 2, 2, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1, name='output', activation='tanh'))
model.summary()


# In[10]:

# Compile model with adam optimizer and learning rate of .0001
adam = Adam(lr=0.0001)
model.compile(loss='mse',optimizer=adam)

# Train model for 20 epochs and a batch size of 128
backend.get_session().run(tf.initialize_all_variables())
model.fit_generator(gen_batches(X_train, y_train, FLAGS.batch_size),
                    len(X_train),FLAGS.num_epochs,
                    validation_data=gen_batches(X_val, y_val, FLAGS.batch_size),
                    nb_val_samples=len(X_val))


# In[19]:

'''
### Add recovery data
# Get steering angles for recovery driving
recovery_angles = pd.read_csv('driving_log_recover.csv', header = None)
recovery_angles.columns = ('Center Image','Left Image','Right Image','Steering Angle','Throttle','Brake','Speed')
recovery_angles = np.array(recovery_angles['Steering Angle'])
'''


# In[20]:

'''
# Construct array for recovery driving images
recovery_images = np.asarray(os.listdir("../IMG_recover/"))
recovery = np.ndarray(shape=(len(recovery_angles), 32, 16, 3))

# Populate recovery driving dataset
count = 0
for image in recovery_images:
    image_file = os.path.join('../IMG_recover', image)
    image_data = ndimage.imread(image_file).astype(np.float32)
    #image_data=cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
    recovery[count] = imresize(image_data, (32,64,3))[12:,:,:]
    count += 1
'''


# ## Step 4: Save model Architecture

# In[11]:

json = model.to_json()
model.save_weights('model_v3.h5')
with open('model_v3.json', 'w') as f:
    f.write(json)


# In[ ]:



