#helpful code: https://gurus.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/
# import the necessary packages
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import datasets
from skimage import exposure
import keras
from keras.datasets import fashion_mnist
import numpy as np
import imutils
import cv2
import sklearn
import time
import psutil
from sklearn.model_selection import StratifiedShuffleSplit

start_time=time.time()

batch_size = 128
num_classes = 10
epochs = 12

# load the MNIST digits dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#format data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("Training matrix shape", x_train.shape)
print("Testing matrix shape", x_test.shape)

results=[]
list1=[]
list2=[]
list3=[]

count_dict={1:0,2:0,3:0,4:0}

for q in range(0,3):
	rng = [2,3,4,5,6,7,8,9] #limit is 5000: 50000 train, 10000 valid
	for i in rng:
		################ CREATING TRAIN AND VAL SETS #################

		print("########################")

		#TRAINING DATA: Creating i instances of each class, VAL DATA: taking 20% of Train
		num_train=i*10
		if(i>=5):
			num_test=int(0.2*num_train)
		else:
			num_test=10

		train_sss = StratifiedShuffleSplit(n_splits=5,
                             test_size=num_test, train_size=num_train, random_state=0)
		for train_index, test_index in train_sss.split(x_train, y_train):
			tempx_train, tempx_val = x_train[train_index], x_train[test_index]
			tempy_train, tempy_val = y_train[train_index], y_train[test_index]
			print("size of kNN balanced training set:", len(tempx_train))
			print("size of kNN balanced val set:", len(tempx_val))

			#verifying correct numbers
			print(tempx_train.shape[0], 'train samples')
			print(tempx_train.shape)
			print(tempx_val.shape[0], 'valid samples')
			print(x_test.shape[0], 'test samples')
			##################################################################


			##################### FINDING THE BEST VALUE OF K ################
			# initialize the values of k for our k-Nearest Neighbor classifier along with the
			# list of accuracies for each value of k
			kVals=range(1, 5)
			accuracies = []

			# loop over various values of `k` for the k-Nearest Neighbor classifier
			for k in range(1, 5):
				# train the k-Nearest Neighbor classifier with the current value of `k`
				model = KNeighborsClassifier(n_neighbors=k)
				model.fit(tempx_train, tempy_train)

				score = model.score(tempx_val, tempy_val) #is this right?  Using val to find best # of neighbors?
				print("k=%d, accuracy=%.2f%%" % (k, score * 100))
				accuracies.append(score)

			# find the value of k that has the largest accuracy
			j = int(np.argmax(accuracies))
			print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[j],accuracies[j] * 100))

			for val in count_dict.keys():
				if(val==kVals[j]):
					count_dict[val]=count_dict[val]+1
			print(count_dict)

			# re-train our classifier using the best k value and predict the labels of the
			# test data
			model = KNeighborsClassifier(n_neighbors=kVals[j])
			model.fit(tempx_train, tempy_train)
			predictions = model.predict(x_test)
			###########################################################################################

			# show a final classification report demonstrating the accuracy of the classifier
			# for each of the digits
			print("EVALUATION ON TESTING DATA FOR", num_train, "TRAINING DATA POINTS")
			print(classification_report(y_test, predictions, digits=3))
			acc=accuracy_score(y_test, predictions)
			print("accuracy: ", acc)
			print("----------------------------")
			results=[i,acc]
			if(q==0):
				list1.append(results)
			elif(q==1):
				list2.append(results)
			else:
				list3.append(results)

print(list1)
print(list2)
print(list3)

datapoints=len(list1)
######finding the variance of three tests#######
variance=np.zeros((datapoints,2))
each_pt=np.zeros((3,2))
for num in range(0,datapoints):
	variance[num][0]=list1[num][0]
	each_pt[0][0]=list1[num][0]
	each_pt[0][1]=list1[num][1]
	each_pt[1][0]=list2[num][0]
	each_pt[1][1]=list2[num][1]
	each_pt[2][0]=list3[num][0]
	each_pt[2][1]=list3[num][1]
	each_var=np.var(each_pt,axis=0)
	variance[num][1]=each_var[1]
print(variance)

######finding the average of three tests########
overall=np.zeros((datapoints,2))
for num in range(0,datapoints):
	overall[num][0]=list1[num][0]
	overall[num][1]=list1[num][1]+list2[num][1]+list3[num][1]
for num in range(0,datapoints):
	overall[num][1]=(overall[num][1])/3.0

print(overall)

######finding the change in accuracy############
acc_change=np.zeros((datapoints-1,2))
for num in range(1,datapoints):
	acc_change[num-1][0]=num-1
	acc_change[num-1][1]=overall[num][1]-overall[num-1][1]
print(acc_change)

np.savetxt("kNN_VariableTrainSet.csv",overall,delimiter=",")
np.savetxt("kNN_Variance.csv",variance,delimiter=",")
np.savetxt("kNN_AccuracyChange.csv",acc_change,delimiter=",")

print("Total Time Elapsed: ", round(time.time()-start_time,1), "seconds")
print(psutil.virtual_memory())
