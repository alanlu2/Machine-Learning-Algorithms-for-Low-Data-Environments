from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn
from sklearn import cross_validation, datasets
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression #same as LinearSVC?

import numpy as np
import time
import psutil

start_time=time.time()

batch_size = 128
num_classes = 10
epochs = 100

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

times=[]

############### CNN MODEL ###########################
def cnnmodel():
    cnnmodel = Sequential()
    cnnmodel.add(Conv2D(32, kernel_size=(3, 3),
             activation='relu',
             input_shape=input_shape))
    cnnmodel.add(Conv2D(64, (3, 3), activation='relu'))
    cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
    cnnmodel.add(Dropout(0.25))
    cnnmodel.add(Flatten())
    cnnmodel.add(Dense(128, activation='relu'))
    cnnmodel.add(Dropout(0.5))
    cnnmodel.add(Dense(num_classes, activation='softmax'))
    cnnmodel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
          metrics=['accuracy'])
    return cnnmodel

############### DNN MODEL ###########################
def dnnmodel():
    dnnmodel = Sequential()
    dnnmodel.add(Dense(512, input_shape=(784,)))
    dnnmodel.add(Activation('relu'))
    dnnmodel.add(Dropout(0.2))
    dnnmodel.add(Dense(512))
    dnnmodel.add(Activation('relu'))
    dnnmodel.add(Dropout(0.2))
    dnnmodel.add(Dense(10))
    dnnmodel.add(Activation('softmax'))
    dnnmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return dnnmodel

############### kNN MODEL ###########################
def knnmodel():
    knnmodel = KNeighborsClassifier(n_neighbors=1)
    return knnmodel

############### RF MODEL ############################
def rfmodel():

    rfmodel=RandomForestClassifier(n_estimators=100)
    return rfmodel

#we want to run the same data on different algorithms
for algorithm in ["knn"]: #ensure this order #,"rf","lin","svm","cnn","dnn"
	alg_time=time.time()

	if (algorithm == "cnn"):
		#reshape everything for cnn after svm, knn, rf
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

		# convert class vectors to binary class matrices - only need to do this once
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)
	else:

		#reshape when back to dnn
		x_train = x_train.reshape(60000, 784)
		x_test = x_test.reshape(10000, 784)

	#CREATING DICTIONARY OF ALGORITHMS
	skmodel_dict={"knn":knnmodel(), "rf":rfmodel(),
                    "lin":LogisticRegression(), "svm": SVC()}
	NNmodel_dict={"cnn":cnnmodel(), "dnn":dnnmodel()}

	#MAIN TRAIN FUNCTION
	def train(alg_str, tempx_train, tempy_train, tempx_val, tempy_val, x_test, y_test):
		#print("Algorithm:", alg_str)
		#print("tempx_train shape:", tempx_train.shape)
		#print("tempy_train shape:", tempy_train.shape)

		if(alg_str!="cnn" and alg_str!="dnn"):
			if(alg_str=="knn"):
				kVals=range(1, 5)
				accuracies_knn = []

				# loop over various values of `k` for the k-Nearest Neighbor classifier
				for k in range(1, 5):
					# train the k-Nearest Neighbor classifier with the current value of `k`
					knnmodel = KNeighborsClassifier(n_neighbors=k)
					knnmodel.fit(tempx_train, tempy_train)

					score = knnmodel.score(tempx_val, tempy_val)
					print("k=%d, accuracy=%.2f%%" % (k, score * 100))
					accuracies_knn.append(score)

					# find the value of k that has the largest accuracy
					j = int(np.argmax(accuracies_knn))
					print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[j],accuracies_knn[j] * 100))

				knnmodel = KNeighborsClassifier(n_neighbors=kVals[j])
				model=knnmodel

			elif(alg_str=="rf"):
				kVals=[100]
				accuracies_rf = []

				# loop over various values of `k` for the Random Forest classifier
				for k in [100]:
					# train the Random Forest classifier with the current value of `k`
					model = RandomForestClassifier(n_estimators=k)
					model.fit(tempx_train, tempy_train)
					score = model.score(tempx_val, tempy_val) #is this right?  Using val to find best # of trees?
					print("k=%d, accuracy=%.2f%%" % (k, score * 100))
					accuracies_rf.append(score)

					# find the value of k that has the largest accuracy
					j = int(np.argmax(accuracies_rf))
					print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[j],accuracies_rf[j] * 100))
				rfmodel=RandomForestClassifier(n_estimators=kVals[j])
				model=rfmodel
			else:
				#svm or lin case
				model=skmodel_dict[alg_str]

			model.fit(tempx_train, tempy_train)
			predictions = model.predict(x_test)
			acc=accuracy_score(y_test, predictions)
			print(alg_str+" accuracy: ", acc)

			#recording the variance
			variance.append([i,acc]) #format: [[5,0.123],[5,0.135],[5,n], etc.]

			return round(acc,4)

		else: #cnn or dnn case
			if(alg_str=="cnn"):
				#callbacks
				filepath='cnnbestweights.hdf5'
				callbacks = [
					EarlyStopping(monitor='val_loss', patience=20, verbose=1),
					ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
					]
				model=cnnmodel()
			if(alg_str=="dnn"):
				#callbacks
				filepath='dnnbestweights.hdf5'
				callbacks = [
					EarlyStopping(monitor='val_loss', patience=20, verbose=1),
					ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
    				]
				model=dnnmodel()
			#fitting the model
			model.fit(tempx_train, tempy_train,
		  		batch_size=batch_size,
		  		epochs=epochs,
		  		verbose=1,
				callbacks=callbacks,
				validation_data=(tempx_val, tempy_val))

			#testing the model on the testing set
			testscore = model.evaluate(x_test, y_test, verbose=0)
			print("testscore: ",testscore[1])

			#recording the variance
			variance.append([i,testscore[1]]) #format: [[5,0.123],[5,0.135],[5,n], etc.]

			return round(testscore[1],4)


	results=[]
	average_data=[]
	variance_data=[]

	#TRAINING DATA: Creating i instances of each class, VAL DATA: taking 20% of Train
	#rng = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
	rng = [1,2,3,4,5,6,7,8,9]    #you can use this line to test
	for i in rng:

		totalscore=0
		variance=[]
		counter=0

		num_train=i*10
		if(i>=5):
			num_test=int(0.2*num_train)
		else:
			num_test=10

		total_iterations=int(6000/(num_train+num_test))
		total_iterations=3

		print("EVALUATION ON TESTING DATA FOR", num_train, "TRAINING DATA POINTS")
		################## TRAINING THE MODEL ######################
		train_sss = StratifiedShuffleSplit(n_splits=total_iterations,
			     test_size=num_test, train_size=num_train, random_state=0)
		for train_index, test_index in train_sss.split(x_train, y_train):
		    counter=counter+1
		    tempx_train, tempx_val = x_train[train_index], x_train[test_index]
		    tempy_train, tempy_val = y_train[train_index], y_train[test_index]
		    print("size of balanced training set:", len(tempx_train))
		    print("size of balanced val set:", len(tempx_val))
		    print("########################")
		    print("ITERATION:",counter,"/",total_iterations)
		    ###################### TESTING ##################
		    accuracy=train(algorithm, tempx_train, tempy_train, tempx_val, tempy_val, x_test, y_test)
		    totalscore=totalscore+accuracy



		average=float(totalscore/total_iterations)
		results=[i,round(average,4)]
		average_data.append(results)

		variance=np.asarray(variance)
		variance_result=np.var(variance,axis=0)
		variance_data.append([i,variance_result[1]])

	print("DATA FOR:", algorithm)
	average_data=np.asarray(average_data)
	print(average_data)

	variance_data=np.asarray(variance_data)
	print(variance_data)

	datapoints=len(average_data)

	######finding the change in accuracy############
	acc_change_data=np.zeros((datapoints-1,2))
	for num in range(1,datapoints):
		acc_change_data[num-1][0]=num-1
		acc_change_data[num-1][1]=average_data[num][1]-average_data[num-1][1]
	print(acc_change_data)

	np.savetxt(("Final_VariableTrainSet_"+algorithm+".csv"),average_data,delimiter=",")
	np.savetxt(("Final_Variance_"+algorithm+".csv"),variance_data,delimiter=",")
	np.savetxt(("Final_AccuracyChange_"+algorithm+".csv"),acc_change_data,delimiter=",")

	print(algorithm,"Time Elapsed: ", round(time.time()-alg_time,1), "seconds")
	times.append(round(time.time()-alg_time,1))
	np.savetxt(("Final_Times.csv"),times,delimiter=",")

print("times of algorithms:", times)
print("Total Time Elapsed: ", round(time.time()-start_time,1), "seconds")
print(psutil.virtual_memory())
