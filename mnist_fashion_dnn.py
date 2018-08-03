#helpful code: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
'''Trains a simple 3-layer DNN on the fashion-MNIST dataset.
12 epochs, batch size is 128.  Lot of room for tuning.
'''

from __future__ import print_function
import keras
import sklearn
import numpy as np
import time
import psutil
from random import shuffle
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit

start_time=time.time()

batch_size = 128
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#format data
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape, 'starting train samples')
print(x_test.shape, 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

results=[]
overall=[]
overall_variance=[]
total_iterations=3

rng = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100] #limit is 5000: 50000 train, 10000 valid
for i in rng:

	################ CREATING TRAIN AND VAL SETS #################

	totalscore_iter=0
	variance=[]

	for iteration in range(0,total_iterations): #the average of a number of iterations


		counter=0
		totalscore_splits=0

		#TRAINING DATA: Creating i instances of each class, VAL DATA: taking 20% of Train
		num_train=i*10
		if(i>=5):
			num_test=int(0.2*num_train)
		else:
			num_test=10

		total_iterations=int(60000/(num_train+num_test))


		train_sss = StratifiedShuffleSplit(n_splits=total_iterations,
		             test_size=num_test, train_size=num_train, random_state=0)
		for train_index, test_index in train_sss.split(x_train, y_train):
			counter=counter+1
			tempx_train, tempx_val = x_train[train_index], x_train[test_index]
			tempy_train, tempy_val = y_train[train_index], y_train[test_index]
			print("size of DNN balanced training set:", len(tempx_train))
			print("size of DNN balanced val set:", len(tempx_val))
			print("------------------")
			print("ITERATION:",counter,"/",total_iterations)
			print("########################")


	    		################## TRAINING THE MODEL ######################
			filepath='dnnbestweights.hdf5'
			callbacks = [
	    		    EarlyStopping(monitor='val_loss', patience=20, verbose=1),
	    		    ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
	    		]

			model = Sequential()
			model.add(Dense(512, input_shape=(784,)))
			model.add(Activation('relu'))
			model.add(Dropout(0.2))
			model.add(Dense(512))
			model.add(Activation('relu'))
			model.add(Dropout(0.2))
			model.add(Dense(10))
			model.add(Activation('softmax'))

			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			#metrics=['accuracy'] displays the validation accuracy
			model.fit(tempx_train, tempy_train,batch_size=batch_size,epochs=epochs,verbose=1,callbacks=callbacks,validation_data=(tempx_val,tempy_val))

				#testing the model on the testing set
			testscore = model.evaluate(x_test, y_test, verbose=0)
			print("testscore: ",testscore[1])

			totalscore_splits=totalscore_splits+testscore[1]

		totalscore_splits=float(totalscore_splits/total_iterations)
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

np.savetxt("DeepNN_VariableTrainSet.csv",overall,delimiter=",")
np.savetxt("DeepNN_Variance.csv",overall_variance,delimiter=",")
np.savetxt("DeepNN_AccuracyChange.csv",acc_change,delimiter=",")

print("Total Time Elapsed: ", round(time.time()-start_time,1), "seconds")
print(psutil.virtual_memory())
