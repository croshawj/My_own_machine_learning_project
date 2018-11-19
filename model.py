
import h5py
import keras
import keras.backend as k 

from keras.layers import Dense, Dropout, Flatten 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Activation

from keras.utils import np_utils # used to split labels to one hot action

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger

import pandas as pd 

def define_model(): # Let's try 2 convs then an output
	model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = 5, padding = 'same', input_shape=(28,28,1), activation = 'relu' ))
	model.add(MaxPooling2D(pool_size = (2,2),strides=(2,2)))
	model.add(Conv2D(filters = 70, kernel_size = 3, padding = 'same',activation = 'relu'))
	model.add(Conv2D(filters = 500, kernel_size = 3, padding = 'same',activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))
	model.add(Conv2D(filters = 1024,kernel_size = 3, padding = 'valid',activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.3))

	model.add(Dense(units = 10))
	model.add(Activation('softmax'))
	
	return model

def dice_coef(y_true,y_pred,smooth = 1): # known as sorensen-dice coefficient.  It gives a measure of how much each data set matches another.
	y_true_f = k.flatten(y_true)
	y_pred_f = k.flatten(y_pred)
	intersection = k.sum(y_true_f*y_pred_f)
	output = (2. * intersection + smooth)/(k.sum(y_true_f) + k.sum(y_pred_f) + smooth)
	return output

def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)


# it is sometimes important to modify the learning rate of the system.  Here are two popular ones.  Others include constant lr or time based decay on lr
def lr_exp_decay(epoch):
	initial_lrate = 0.01
	drop = 1
	lrate = initial_lrate * math.pow(drop,epoch)
	return lrate

def lr_step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
	return lrate



#load data files.  Apply one hot action to test and train labels
h5f = h5py.File('G:\\My Drive\\University_of_Alberta\\RA\\My_own_machine_learning_project\\data\\mnist_data.h5','r')
x_train = h5f['images_train']
y_train = h5f['labels_train'] 
x_test = h5f['images_test']
y_test = h5f['labels_test']
x_validate = h5f['images_validate']
y_validate = h5f['labels_validate']
h5f.close
y_train = np_utils.to_categorical(y_train).astype('int32')
y_test = np_utils.to_categorical(y_test).astype('int32')
y_validate = np_utils.to_categorical(y_validate).astype('int32')

#print(y_train.shape)

#list callbacks 

model_name = 'myveryfirstmodel'
lr_rate_callback = LearningRateScheduler(lr_exp_decay,verbose = 1)
model_checkpoint = ModelCheckpoint(".\\checkpoints\\weights.{epoch:02d}-{val_loss:.2f}-{val_dice_coef:.2f}.hdf5", monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'min', period = 10)
csv_callback = CSVLogger(".\\checkpoints\\{}.csv".format(model_name), separator = ',', append =False)
callbacks = [lr_rate_callback, model_checkpoint, csv_callback]


model = define_model()
model.compile(loss = dice_coef_loss, optimizer = 'adam', metrics=['accuracy',dice_coef]) #loss = 'categorical_crossentropy'
print(model.summary())
#model.fit(x = x_train, y = y_train, batch_size = 32, epochs = 12, verbose = 1, callbacks = None, validation_split = 0.0, validation_data = None, shuffle = 'batch')
#model.save(model_name + '.h5')
