# -*- coding: utf-8 -*-

#%%
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# plot import matplotlib
import matplotlib
import matplotlib.pyplot as plt

#%%

# batch size
batch_size = 128

# no. of object classes
no_classes = 10

# no. of epochs
no_epochs = 12

#input image dimensions
img_rows, img_cols = 28, 28

# no. of filters to use
no_filters = 32

#size of pooling area for maxpooling
no_pool = 2
pool_size = (2, 2)

#convolution kernel size
no_conv = 3



#%%

# import the data shuffled between train and test

print(K.image_dim_ordering())

(x_train, y_train), (x_test, y_test)=mnist.load_data()
print (x_train.shape)
print (x_train.shape[0])
#x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
#x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
#input_shape = (1,img_rows, img_cols) 

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
input_shape = (img_rows, img_cols, 1) 

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'x_train sample')
print(x_test.shape[0], 'x_test sample')

#convert class vectors to binary class matrices
y_train=np_utils.to_categorical(y_train, no_classes)
y_test=np_utils.to_categorical(y_test, no_classes)

i=4600

#plt.imshow(x_train[i,0], interpolation='nearest')
#plt.imshow(x_train[i,0], extent=[1,1,1,1], aspect='auto')
plt.imshow(x_train[i,:,:,0], interpolation='nearest')
print (x_train.shape[3])
print('y_label', y_train[i,:])

#%%
# define CNN model

model = Sequential()

model.add(Convolution2D(no_filters, no_conv, no_conv, border_mode='valid', input_shape = (img_rows, img_cols,1)))
convol1=Activation('relu')
model.add(convol1)
model.add(Convolution2D(no_filters, no_conv, no_conv))
convol2=Activation('relu')
model.add(convol2)
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(no_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=batch_size, nb_epoch = no_epochs, show_accuracy=True, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])
print(model.predict_classes(x_test[1:5]))
print(y_test[1:5])
