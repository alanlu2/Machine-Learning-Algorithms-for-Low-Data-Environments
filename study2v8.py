'''
THIS STUDY LOOKS AT USING A VARIABLE TRAIN DATASET TO OPTIMIZE CLASSIFICATION ACCURACY ON FASHION_MNIST.  BALANCED, NEIL'S MODEL
'''

from __future__ import print_function
import keras
import numpy as np
import time
import random
import functools
import operator as op
from itertools import combinations
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, subtract
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

start_time=time.time()

batch_size = 15
epochs = 40
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
total_iterations = 5

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

def ncr(n, r):
    r = min(r, n-r)
    numer = functools.reduce(op.mul, range(n, n-r, -1), 1)
    denom = functools.reduce(op.mul, range(1, r+1), 1)
    return int(numer//denom)

########################### PREPROCESSING DATA #################################################
def get_pairs(X,y,num1,num2):

    train_bins = [[] for _ in range(10)]

    #each bin has same-label images - train_bin[0] has all images with label 0, etc.
    for idx, label in enumerate(y):
        train_bins[label].append(X[idx])
    train_bins = np.array(train_bins)


    #Now we have 6000 images in train_bins[0], train_bins[1], etc.
    print("# of images in each bin:",len(train_bins[1]))

    #testing set sampled from unused training set
    first_images=[]
    first_labels=[]
    second_images=[]
    second_labels=[]

    #choosing 2000 images from first half, putting 30000 images in second half
    #testing set is the unused set from first half
    for i in range(0,10):
        for img1 in train_bins[i][:num1]:
            first_images.append(img1)
        for img2 in train_bins[i][len(train_bins[i])-num2:]:
            second_images.append(img2)

    #assigning the corresponding labels
    for i in range(0,10):
        for j in range(0,num1):
            first_labels.append(i)
        for j in range(0,num2):
            second_labels.append(i)

    print("# of images chosen from first half:",len(first_images))
    print("# of labels chosen from first half:",len(first_labels))
    print("# of images in the second half:",len(second_images))
    print("# of labels in the second half:",len(second_labels))


    #define variables for indices and pairs
    sim_idxs=[]
    dif_idxs=[]
    sim_pairs=[]
    dif_pairs=[]

    #creating all possible pairs
    comb=combinations(range(num1*10),2)

    #distiguishing the indices of the similar and different pairs
    for i in list(comb):
        if(first_labels[i[0]]==first_labels[i[1]]):
            sim_idxs.append(i)
        else:
            dif_idxs.append(i)

    #put the images in sim_pairs, dif_pairs
    for elem in sim_idxs:
        sim_pairs.append(first_images[elem[0]].reshape((28,28,1)))
        sim_pairs.append(first_images[elem[1]].reshape((28,28,1)))
    for elem in dif_idxs:
        dif_pairs.append(first_images[elem[0]].reshape((28,28,1)))
        dif_pairs.append(first_images[elem[1]].reshape((28,28,1)))

    print("# of similar images:",len(sim_pairs))
    print("# of different images:",len(dif_pairs))

    #reshape images to fit the input of the siamese network
    sim_pairs=np.array(sim_pairs)
    sim_pairs=sim_pairs.reshape((len(sim_idxs),2,28,28,1))
    dif_pairs=np.array(dif_pairs)
    dif_pairs=dif_pairs.reshape((len(dif_idxs),2,28,28,1))

    #defining new labels
    new_sim_labels=[1]*len(sim_pairs)
    new_dif_labels=[0]*len(dif_pairs)

    #concatenate the similar and different pairs together
    all_pairs=np.concatenate((sim_pairs,dif_pairs))
    all_labels=new_sim_labels+new_dif_labels


    print("...balancing for 50-50 baseline")
    #BALANCING THE TRAINING SET
    all_pairs=all_pairs[:len(sim_pairs)*2]
    all_labels=all_labels[:len(all_pairs)]
    all_pairs.reshape((len(all_pairs),2,28,28,1))
    print(len(all_pairs))

    # Categorical encoding
    pair_labels = keras.utils.to_categorical(all_labels)

    #split up data into left and right columns, prepare for input
    left=all_pairs[:,0,:,:]
    right=all_pairs[:,1,:,:]

    return [left,right], pair_labels, np.array(first_images).reshape(len(first_images),28,28,1), first_labels,np.array(second_images).reshape(len(second_images),28,28,1), second_labels
#############################################################################################
def get_model():
    ######################################## SIAMESE NETWORK ################################
    #CNN part of the SNN
    input = Input(input_shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    out = Flatten()(x)

    #encoded model
    model = Model(input, out)

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    #distance formula and final FC layer
    dist = subtract([encoded_l, encoded_r])
    x = Dense(128, activation='relu')(dist)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    prediction = Dense(2, activation='softmax', name='output')(x)
    siamese_net = Model(inputs=[left_input,right_input], outputs=[prediction])


    siamese_net.compile(loss=["binary_crossentropy"
                              ],optimizer='adam', metrics=['accuracy']
                        )

    return siamese_net, model


##############################################################################################

#most of these variables are for finding the final average
mean_scores=[]
overall_siamese=[]
overall_testing=[]
total_reg_acc_score=0
total_enc_acc_score=0
total_mean=0


first_num_imgs=150 #number of images per class that we choose from the first half
second_num_imgs=3000 #number of images per class in the second half

#number of total, similar, and different pairs
num_total_pairs= ncr(10*first_num_imgs,2)
num_sim_pairs= 10*ncr(first_num_imgs,2)
num_dif_pairs=num_total_pairs-num_sim_pairs




for iteration in range(total_iterations):
    print("ITERATION:",iteration+1,"/",total_iterations)

    #generating train, validation, and hold pairs
    train_images, train_labels, first_images, first_labels, second_images, second_labels = get_pairs(x_train, y_train, first_num_imgs,second_num_imgs)

    val_images, val_labels, first_val_images, first_val_labels, second_val_images, second_val_labels = get_pairs(x_test[:int(len(x_test)/2)], y_test[:int(len(y_test)/2)], 30,3) #300

    hold_images, hold_labels, first_hold_images, first_hold_labels, second_hold_images, second_hold_labels = get_pairs(x_test[int(len(x_test)/2):], y_test[int(len(y_test)/2):], 100,3)


    # defining the callbacks
    filepath='snnbestweights_v8.hdf5'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
        LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch)),
    ]

    ############################ SIAMESE MODEL ###################################
    scores=[]
    testing=[]

    print("Siamese accuracy for",first_num_imgs,"images per class")

    #fitting the snn
    siamese_net, encoded = get_model()
    siamese_net.fit(train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        validation_data=(val_images, val_labels),
        shuffle=True
    ) #sample_weight=sample_weights

    trainscore = siamese_net.evaluate(train_images, train_labels,verbose=1)[1]

    valscore = siamese_net.evaluate(val_images, val_labels,verbose=1)[1]

    holdscore = siamese_net.evaluate(hold_images, hold_labels,verbose=1)[1]

    scores.append([trainscore, valscore, holdscore])
    np.savetxt("snn_network_accuracies_s2_v8_"+str(iteration)+".csv",scores,delimiter=",")

    scores = np.array(scores)
    scores = pd.DataFrame({'Train':scores[:,0], 'Val':scores[:,1], 'Hold':scores[:,2]})
    print('Mean Scores:', scores.mean(axis=0))
    total_mean=total_mean+scores.mean(axis=0)

    #---------------------------------knn model------------------------------
    print("...testing on knn model...")
    enc_first_images=encoded.predict(first_images)
    enc_second_images=encoded.predict(second_images) #x_train


    #reshape the images to fit into the knn
    first_images=first_images.reshape(first_num_imgs*10,784)
    second_images=second_images.reshape(second_num_imgs*10,784)

    #---------------------------------choose best k--------------------------
    kVals=range(1, 5)
    accuracies=[]
    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, 5):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        bestkmodel = KNeighborsClassifier(n_neighbors=k)
        bestkmodel.fit(first_images, first_labels)

        score = bestkmodel.score(first_val_images.reshape(len(first_val_labels),784), first_val_labels)
        accuracies.append(score)

    # find the value of k that has the largest accuracy
    j = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[j],accuracies[j] * 100))
    knnmodel = KNeighborsClassifier(n_neighbors=kVals[j])


    #getting the predictions of the regular and encoded models
    knnmodel.fit(first_images, first_labels)
    reg_predict = knnmodel.predict(second_images)

    knnmodel.fit(enc_first_images, first_labels)
    enc_predict = knnmodel.predict(enc_second_images)

    #getting the accuracies of the regular and encoded models
    reg_acc=accuracy_score(second_labels, reg_predict)
    enc_acc=accuracy_score(second_labels, enc_predict)


    print("Testing accuracy for:",second_num_imgs,"images per class")
    print("reg accuracy: ", reg_acc)
    print("enc accuracy: ", enc_acc)
    total_reg_acc_score=total_reg_acc_score+reg_acc
    total_enc_acc_score=total_enc_acc_score+enc_acc
    testing.append([reg_acc,enc_acc])
    np.savetxt("snn_testing_accuracies_s2_v8_"+str(iteration)+".csv",testing,delimiter=",")
    print("-------------------------------------")

#TAKE THE AVERAGE OF THE ALL OF THE RUNS
avg_of_avgs=total_mean/total_iterations
print("SNN AVG T,V,H:",avg_of_avgs)
overall_siamese.append([first_num_imgs,avg_of_avgs[1],avg_of_avgs[2],avg_of_avgs[0]])

avg_reg_acc_score=total_reg_acc_score/total_iterations
avg_enc_acc_score=total_enc_acc_score/total_iterations
print("TESTING AVG REG,ENC:",avg_reg_acc_score,avg_enc_acc_score)
overall_testing.append([first_num_imgs,avg_reg_acc_score,avg_enc_acc_score])



print("#########################################")
print("FINAL SIAMESE ACCURACIES RESULTS:",overall_siamese)
print("FINAL TESTING ACCURACIES RESULTS:",overall_testing)
print("Total Time Elapsed: ", round(time.time()-start_time,1), "seconds")


np.savetxt("snn_network_accuracies_s2_v8.csv",overall_siamese,delimiter=",")
np.savetxt("snn_testing_accuracies_s2_v8.csv",overall_testing,delimiter=",")
