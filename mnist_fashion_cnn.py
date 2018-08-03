#helpful code: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
#https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
'''Trains a simple CNN on the fashion-MNIST dataset.
12 epochs, batch size is 128.  Lot of room for tuning.
'''

from __future__ import print_function
import keras
import numpy as np
import time
import psutil
import sklearn
from random import shuffle
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit

start_time=time.time()

batch_size = 128
num_classes = 10
epochs = 100
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#format data
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'starting train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

results=[]
overall=[]
overall_variance=[]
total_iterations=3


rng = [1,2] #number of examples per class - in this case 10 and 20
for i in rng:

	################ CREATING TRAIN AND VAL SETS #################

	totalscore_iter=0
	variance=[]

	for iteration in range(0,total_iterations): #the average of a number of iterations
		print("########################")
		print("ITERATION:",iteration)

		#TRAINING DATA: Creating i instances of each class, VAL DATA: taking 20% of Train
		num_train=i*10
		if(i>=5):
			num_test=int(0.2*num_train)
		else:
			num_test=10

		totalscore_splits=0

		train_sss = StratifiedShuffleSplit(n_splits=5,
		             test_size=num_test, train_size=num_train, random_state=0)
		for train_index, test_index in train_sss.split(x_train, y_train):
			tempx_train, tempx_val = x_train[train_index], x_train[test_index]
			tempy_train, tempy_val = y_train[train_index], y_train[test_index]
			print("size of CNN balanced training set:", len(tempx_train))
			print("size of CNN balanced val set:", len(tempx_val))
			print("------------------")


	    		################## TRAINING THE MODEL ######################
			filepath='cnnbestweights.hdf5'
			callbacks = [
	    		    EarlyStopping(monitor='val_loss', patience=20, verbose=1),
	    		    ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
	    		]


			model = Sequential()
			model.add(Conv2D(32, kernel_size=(3, 3),
			     activation='relu',
			     input_shape=input_shape))

			model.add(Conv2D(64, (3, 3), activation='relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))
			model.add(Dropout(0.25))
			model.add(Flatten())

			model.add(Dense(128, activation='relu'))

			model.add(Dropout(0.5))
			model.add(Dense(num_classes, activation='softmax'))

			model.compile(loss=keras.losses.categorical_crossentropy,
			      optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])
	    		#fitting the model to the validation set
			model.fit(tempx_train, tempy_train,batch_size=batch_size,epochs=epochs,verbose=1,callbacks=callbacks,validation_data=(tempx_val,tempy_val))

	    		#testing the model on the testing set
			testscore = model.evaluate(x_test, y_test, verbose=0)
			print("testscore: ",testscore[1])

			totalscore_splits=totalscore_splits+testscore[1]

		totalscore_splits=float(totalscore_splits/5)
		print("totalscore_splits:", totalscore_splits)
		totalscore_iter=totalscore_iter+totalscore_splits

        	#recording the variance
		variance.append([i,testscore[1]]) #format: [[5,0.123],[5,0.135],[5,n], etc.]
	variance=np.asarray(variance)
	variance_result=np.var(variance,axis=0)
	overall_variance.append([i,variance_result[1]])

	#for the average
	average=float(totalscore_iter/total_iterations)
	results=[i,round(average,4)]
	overall.append(results)



overall=np.asarray(overall)
print(overall)

overall_variance=np.asarray(overall_variance)
print(overall_variance)

datapoints=len(overall)

######finding the change in accuracy############
acc_change=np.zeros((datapoints-1,2))
for num in range(1,datapoints):
	acc_change[num-1][0]=num-1
	acc_change[num-1][1]=overall[num][1]-overall[num-1][1]
print(acc_change)

np.savetxt("CNN_VariableTrainSet.csv",overall,delimiter=",")
np.savetxt("CNN_Variance.csv",overall_variance,delimiter=",")
np.savetxt("CNN_AccuracyChange.csv",acc_change,delimiter=",")

print("Total Time Elapsed: ", round(time.time()-start_time,1), "seconds")
print(psutil.virtual_memory())
