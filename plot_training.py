import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

'''
We can use 'agg' tp save files remotely over ssh
https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab/4706614#4706614
'''

#plt.switch_backend('agg')



#style of the plot
plt.style.use('ggplot')

#define the learning rate function that is used for training in order to plot it as a function number of epochs

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

df = pd.read_csv('filelocation!!!')
df['lr'] = df['epoch'].apply(lr_decay)

fig,ax = plt.subplots(2,3,figsize = (15,6))

ax[0][0].scatter(df['epoch'],df['loss'],color='blue')
ax[0][0].plot(df['epoch'],df['loss'],color='blue')
ax[0][0].set_xlabel('epoch')
ax[0][0].set_ylabel('loss')
    
ax[0][1].scatter(df['epoch'],df['dice_coef'],color='red')
ax[0][1].plot(df['epoch'],df['dice_coef'],color='red')
ax[0][1].set_xlabel('epoch')
ax[0][1].set_ylabel('dice_coef')
    
ax[0][2].scatter(df['epoch'],df['lr'],color='green')
ax[0][2].plot(df['epoch'],df['lr'],color='green')
ax[0][2].set_xlabel('epoch')
ax[0][2].set_ylabel('learning rate')
ax[0][2].set_yscale('log')
    
ax[1][0].scatter(df['epoch'],df['val_loss'],color='blue')
ax[1][0].plot(df['epoch'],df['val_loss'],color='blue')
ax[1][0].set_xlabel('epoch')
ax[1][0].set_ylabel('val_loss')
    
ax[1][1].scatter(df['epoch'],df['val_dice_coef'],color='red')
ax[1][1].plot(df['epoch'],df['val_dice_coef'],color='red')
ax[1][1].set_xlabel('epoch')
ax[1][1].set_ylabel('val_dice_coef')
    
ax[1][2].scatter(df['epoch'],df['val_acc'],color='green')
ax[1][2].plot(df['epoch'],df['val_acc'],color='green')
ax[1][2].set_xlabel('epoch')
ax[1][2].set_ylabel('val_acc')


plt.subplots_adjust(wspace = 0.3,hspace = 0.3)

plt.show()