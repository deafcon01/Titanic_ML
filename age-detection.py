#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


import os
import random
import pandas as pd
from imageio import imread


# In[3]:


DATUMS_PATH=os.getenv('DATUMS_PATH')

DATASET_NAME=os.getenv('DATASET_NAME')
#DATASET_NAME
DATUMS_PATH


# In[4]:


get_ipython().run_line_magic('pwd', '')


# In[5]:


train = pd.read_csv(os.path.join(DATUMS_PATH,DATASET_NAME,'train.csv'))


# In[6]:


test = pd.read_csv(os.path.join(DATUMS_PATH,DATASET_NAME,'test.csv'))


# In[7]:


#randomly choose image and print
i = random.choice(train.index)
image_name = train.ID[i]
image = imread(os.path.join(DATUMS_PATH,DATASET_NAME,'Train',image_name))
print('Age :',train.Class[i])


# In[8]:


imshow(image)


# In[9]:


from skimage.transform import resize


# In[10]:


import numpy as np
temp = []
for image_name in train.ID:
    image_path = os.path.join(DATUMS_PATH,DATASET_NAME,'Train',image_name)
    image = imread(image_path)
    image = resize(image,(32,32))
    image = image.astype('float32')
    temp.append(image)
train_X= np.stack(temp)


# In[11]:


temp = []
for img_name in test.ID:
    img_path = os.path.join(DATUMS_PATH,DATASET_NAME, 'Test', img_name)
    img = imread(img_path)
    img = resize(img, (32, 32))
    temp.append(img.astype('float32'))

test_x = np.stack(temp)


# In[12]:


#Normalize images
train_x = train_X / 255.
test_x = test_x / 255.


# In[13]:


train.Class.value_counts(normalize=True)


# In[14]:


import keras
from sklearn.preprocessing import LabelEncoder


# In[15]:


lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)


# In[16]:


input_num_units = (32,32,3)
hidden_num_units = 500
output_num_units = 3
epoch = 5
batch_size = 128


# In[17]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer


# In[18]:


model = Sequential([
    InputLayer(input_shape = input_num_units),
    Flatten(),
    Dense(units=hidden_num_units,activation='relu'),
    Dense(units=output_num_units,activation='softmax')
])


# In[19]:


model.summary()


# In[20]:


model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=batch_size,epochs=epoch)


# In[21]:


model.fit(train_x, train_y, batch_size=batch_size,epochs=epoch,verbose=1, validation_split=0.2)


# In[22]:


import tensorflow as tf


# In[23]:


model_new=tf.keras.Sequential([
    tf.keras.layers.Conv2D(50,(3,3),activation='relu',input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(100,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(100,(3,3),activation='relu'),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(hidden_num_units,activation='relu'),
    tf.keras.layers.Dense(output_num_units,activation='softmax'),
])


# In[24]:


model_new.summary()


# In[25]:


model_new.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model_new.fit(train_x,train_y,batch_size=batch_size,epochs=epoch)


# In[26]:


model_new.fit(train_x, train_y, batch_size=batch_size,epochs=epoch,verbose=1, validation_split=0.2)


# In[31]:


pred = model_new.predict_classes(test_x)
pred = lb.inverse_transform(pred)
test['Class'] = pred
test.to_csv(os.path.join(DATUMS_PATH,DATASET_NAME,'sub.csv'),index=False)

