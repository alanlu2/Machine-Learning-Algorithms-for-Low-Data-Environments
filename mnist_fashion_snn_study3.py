'''
THIS STUDY LOOKS AT THE GENERALIZATION OF TRAINING ON THE MNIST DATASET TO CLASSIFY IMAGES IN THE FASHION_MNIST DATASET, UNBALANCED, SIMPLE MODEL
'''


from __future__ import print_function
import keras
import numpy as np
import time
import random
import functools
import operator as op
from itertools import combinations
from keras.datasets import mnist,fashion_mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, subtract
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

import data_augmentation

start_time=time.time()

batch_size = 15
epochs = 40
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
total_iterations=8

#function to define n choose r
def ncr(n, r):
    r = min(r, n-r)
    numer = functools.reduce(op.mul, range(n, n-r, -1), 1)
    denom = functools.reduce(op.mul, range(1, r+1), 1)
    return int(numer//denom)



def get_pairs(X, y, n):
    #train_bins is a list of 10 empty lists
    train_bins = [[] for _ in range(10)]

    #each bin has same-label images - train_bin[0] has all images with label 0, etc.
    for idx, label in enumerate(y):
        train_bins[label].append(X[idx])
    train_bins = np.array(train_bins)

    #Now we have 6000 images in train_bins[0], train_bins[1], etc.
    images=[]
    labels=[]

    #making list of 30 images and labels - 3 from each class
    for idx in range(0,len(train_bins)):
        for i in range(0,n):
            images.append(random.choice(train_bins[idx]))
            labels.append(idx)

    #define variables for indices and pairs
    sim_idxs=[]
    dif_idxs=[]
    sim_pairs=[]
    dif_pairs=[]

    #creating all possible pairs
    comb=combinations(range(n*len(train_bins)),2)

    #distiguishing the indices of the similar and different pairs
    for i in list(comb):
        if(labels[i[0]]==labels[i[1]]):
            sim_idxs.append(i)
        else:
            dif_idxs.append(i)

    #put the images in sim_pairs, dif_pairs
    for elem in sim_idxs:
        sim_pairs.append(images[elem[0]].reshape((28,28,1)))
        sim_pairs.append(images[elem[1]].reshape((28,28,1)))
    for elem in dif_idxs:
        dif_pairs.append(images[elem[0]].reshape((28,28,1)))
        dif_pairs.append(images[elem[1]].reshape((28,28,1)))

    #reshape images to fit the input of the siamese network
    sim_pairs=np.array(sim_pairs)
    sim_pairs=sim_pairs.reshape((len(sim_idxs),2,28,28,1))
    left_sim_pair=sim_pairs[:,0,:,:]
    right_sim_pair=sim_pairs[:,1,:,:]

    dif_pairs=np.array(dif_pairs)
    dif_pairs=dif_pairs.reshape((len(dif_idxs),2,28,28,1))

    #defining new labels
    new_sim_labels=[1]*len(sim_pairs)
    new_dif_labels=[0]*len(dif_pairs)

    #concatenate the similar and different pairs together
    all_pairs=np.concatenate((sim_pairs,dif_pairs))
    all_labels=new_sim_labels+new_dif_labels

    '''
    #BALANCING THE TRAINING SET
    all_pairs=all_pairs[:len(sim_pairs)*2]
    all_labels=all_labels[:len(all_pairs)]
    all_pairs.reshape((len(all_pairs),2,28,28,1))
    '''

    # Categorical encoding
    pair_labels = keras.utils.to_categorical(all_labels)
    sim_labels = keras.utils.to_categorical(new_sim_labels)

    #split up data into left and right columns, prepare for input
    left=all_pairs[:,0,:,:]
    right=all_pairs[:,1,:,:]


    return [left, right], pair_labels, np.array(images).reshape(len(images),28,28,1), labels, [left_sim_pair,right_sim_pair], sim_labels

#tune the dropout to adjust hyperparameters
dropout=0.7
def get_model():
    ######################################## SIAMESE NETWORK ################################

    #CNN part of the SNN
    input = Input(input_shape)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(input)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    out = Flatten()(x)

    #encoded model
    enc_model = Model(input, out)

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    encoded_l = enc_model(left_input)
    encoded_r = enc_model(right_input)

    #distance formula and final FC layer
    dist = subtract([encoded_l, encoded_r])
    x = Dense(32, activation='relu')(dist)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation='relu')(x)
    prediction = Dense(2, activation='softmax')(x)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    siamese_net.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    return siamese_net, enc_model

###################################### IMPORTING DATA ##################################
#importing both mnist and fashion mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(fash_x_train, fash_y_train), (fash_x_test, fash_y_test) = fashion_mnist.load_data()

#format fash data
fash_x_train = fash_x_train.reshape(60000, 784)
fash_x_test = fash_x_test.reshape(10000, 784)
fash_x_train = fash_x_train.astype('float32')
fash_x_test = fash_x_test.astype('float32')
fash_x_train /= 255
fash_x_test /= 255
print("Fash Training matrix shape", fash_x_train.shape)
print("Fash Testing matrix shape", fash_x_test.shape)

#format data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("Training matrix shape", x_train.shape)
print("Testing matrix shape", x_test.shape)


mean_scores=[]
overall_siamese=[]
overall_testing=[]

#number of images per class used
rng=[2,3,4,5,6,7,8,9,10]
counter=0
for choose in rng:

    #most of these variables are used for taking the average
    num_total_pairs= ncr(10*choose,2)
    num_sim_pairs= 10*ncr(choose,2)
    num_dif_pairs=num_total_pairs-num_sim_pairs
    total_reg_acc_score=0
    total_enc_acc_score=0
    best_train=0
    best_val=0
    best_holdout=0
    total_mean=0
    total_sim_mean=0
    for iteration in range(total_iterations):
        print("ITERATION:",iteration+1,"/",total_iterations)

        #generating train, validation, and hold pairs for both mnist and fashion mnist
        train_images, train_labels, orig_tr_images, orig_tr_labels, train_sim_images, train_sim_labels = get_pairs(x_train, y_train, choose)

        val_images, val_labels, orig_val_images, orig_val_labels, val_sim_images, val_sim_labels = get_pairs(x_test[:int(len(x_test)/2)], y_test[:int(len(x_test)/2)], choose)

        hold_images, hold_labels, orig_hold_images, orig_hold_labels, hold_sim_images, hold_sim_labels = get_pairs(x_test[int(len(x_test)/2):], y_test[int(len(x_test)/2):], choose)

        fash_train_images, fash_train_labels, orig_fash_tr_images, orig_fash_tr_labels, fash_train_sim_images, fash_train_sim_labels = get_pairs(fash_x_train, fash_y_train, choose)

        fash_val_images, fash_val_labels, orig_fash_val_images, orig_fash_val_labels, fash_val_sim_images, fash_val_sim_labels = get_pairs(fash_x_test[:int(len(fash_x_test)/2)], fash_y_test[:int(len(fash_x_test)/2)], choose)

        fash_hold_images, fash_hold_labels, orig_fash_hold_images, orig_fash_hold_labels, fash_hold_sim_images, fash_hold_sim_labels = get_pairs(fash_x_test[int(len(fash_x_test)/2):], fash_y_test[int(len(fash_x_test)/2):], choose)

        ######################## TRAINING ##########################

        # defining the callbacks
        filepath='snnbestweights'+str(choose)+'_s3_v1.hdf5'
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
            ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=0),
            LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch)),
        ]


        #------------------------siamese model---------------------------------
        highest_score=0
        val_scores = {}
        scores=[]
        simscores=[]
        save_scores=[]
        testing=[]

        #defining the similar and different weights, necessary for unbalanced model
        sim_weight=float(1/((num_dif_pairs/num_sim_pairs)+1))*float(num_dif_pairs/num_sim_pairs)
        print(sim_weight)
        dif_weight=1-sim_weight
        sample_weights=[sim_weight]*num_sim_pairs+[dif_weight]*num_dif_pairs #num_dif_pairs for unbalanced
        sample_weights=np.array(sample_weights)

        print("Siamese accuracy for:",choose,"images per class")
        #THIS LOOP TRIES TO FIND THE BEST MODEL
        for i in range(2):
            print("MODEL "+str(i+1)+"/2")
            siamese_net, encoded = get_model()
            siamese_net.fit(train_images,
                train_labels,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0,
                validation_data=(val_images, val_labels),
                shuffle=True,
                sample_weight=sample_weights #sample weight
            ) #matthews correlation coefficient

            trainscore = siamese_net.evaluate(train_images, train_labels,verbose=1)[1]

            valscore = siamese_net.evaluate(val_images, val_labels,verbose=1)[1]

            holdscore = siamese_net.evaluate(hold_images, hold_labels,verbose=1)[1]

            sim_trainscore = siamese_net.evaluate(train_sim_images, train_sim_labels,verbose=0)[1]

            sim_valscore = siamese_net.evaluate(val_sim_images, val_sim_labels,verbose=0)[1]

            sim_holdscore = siamese_net.evaluate(hold_sim_images, hold_sim_labels,verbose=0)[1]

            val_scores[i]=valscore

            simscores.append([sim_trainscore,sim_valscore,sim_holdscore])
            scores.append([trainscore, valscore, holdscore])

            #find highest test score
            if(val_scores[i]>=highest_score):
                highest_score=val_scores[i]
                correct_index=i
                #save the models
                snn=siamese_net
                encoded_model=encoded

        #this is the highest score for the best model
        print("highest score",highest_score)

        scores = np.array(scores)
        scores = pd.DataFrame({'Train':scores[:,0], 'Val':scores[:,1], 'Hold':scores[:,2]})
        print(scores)
        print('Mean Scores:', scores.mean(axis=0))
        total_mean=total_mean+scores.mean(axis=0)

        #retrain the model with correct model
        trainscore = snn.evaluate(train_images, train_labels,verbose=0)[1]
        valscore = snn.evaluate(val_images, val_labels,verbose=0)[1]
        holdscore = snn.evaluate(hold_images, hold_labels,verbose=0)[1]

        save_scores.append([choose,iteration,trainscore, valscore, holdscore])
        np.savetxt("snn_network_accuracies_s3_"+str(choose)+"_"+str(iteration)+".csv",save_scores,delimiter=",")


        best_train=best_train+trainscore
        best_val=best_val+valscore
        best_holdout=best_holdout+holdscore

        print("best t,v,h:",trainscore,",", valscore,",", holdscore)

        #we want to track the scores for the similar pairs to know that the model is actually learning something
        print("...")
        simscores = np.array(simscores)
        simscores = pd.DataFrame({'Train':simscores[:,0], 'Val':simscores[:,1], 'Hold':simscores[:,2]})
        print(simscores)
        print('Mean simscores:', simscores.mean(axis=0))
        total_sim_mean=total_sim_mean+simscores.mean(axis=0)

        #transfer learning from mnist snn, now fitting on fashion_mnist data
        snn.fit(fash_train_images,
            fash_train_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
            validation_data=(fash_val_images, fash_val_labels),
            shuffle=True,
            sample_weight=sample_weights #sample weight
        )

        #---------------------------------knn model------------------------------
        print("...testing on knn model...")
        #creating encoded data
        enc_train_imgs=encoded_model.predict(orig_fash_tr_images) #x_train
        enc_hold_imgs=encoded_model.predict(orig_fash_hold_images) #x_test


        #reshape the images to fit into the knn
        orig_fash_tr_images=orig_fash_tr_images.reshape(choose*10,784)
        orig_fash_hold_images=orig_fash_hold_images.reshape(choose*10,784)



        #---------------------------------choose best k--------------------------

        kVals=range(1, 5)
        accuracies=[]
        # loop over various values of `k` for the k-Nearest Neighbor classifier
        for k in range(1, 5):
    		# train the k-Nearest Neighbor classifier with the current value of `k`
            bestkmodel = KNeighborsClassifier(n_neighbors=k)
            bestkmodel.fit(orig_fash_tr_images, orig_fash_tr_labels)

            score = bestkmodel.score(orig_fash_val_images.reshape(choose*10,784), orig_fash_val_labels)
            accuracies.append(score)

    	# find the value of k that has the largest accuracy
        j = int(np.argmax(accuracies))
        print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[j],accuracies[j] * 100))


    	# re-train our classifier using the best k values
        knnmodel = KNeighborsClassifier(n_neighbors=kVals[j])

        #making regular and encoded predictions
        knnmodel.fit(orig_fash_tr_images, orig_fash_tr_labels)
        reg_predict = knnmodel.predict(orig_fash_hold_images)

        knnmodel.fit(enc_train_imgs, orig_fash_tr_labels)
        enc_predict = knnmodel.predict(enc_hold_imgs)

        #getting regular and encoded accuracies
        reg_acc=accuracy_score(orig_fash_hold_labels, reg_predict)
        enc_acc=accuracy_score(orig_fash_hold_labels, enc_predict)

        print("Testing accuracy for:",choose,"images per class")
        print("reg accuracy: ", reg_acc)
        print("enc accuracy: ", enc_acc)
        total_reg_acc_score=total_reg_acc_score+reg_acc
        total_enc_acc_score=total_enc_acc_score+enc_acc
        testing.append([choose,iteration,reg_acc,enc_acc])
        np.savetxt("snn_testing_accuracies_s3_"+str(choose)+"_"+str(iteration)+".csv",testing,delimiter=",")

        print("-------------------------------------")
    #TAKE THE AVERAGE OF THE ALL OF THE RUNS
    avg_of_avgs=total_mean/total_iterations
    print("AVG OF AVGS:",avg_of_avgs)
    avg_of_sim_avgs=total_sim_mean/total_iterations
    print("AVG OF SIM AVGS:",avg_of_sim_avgs)
    print("#######################################")
    avg_best_train=best_train/total_iterations
    avg_best_val=best_val/total_iterations
    avg_best_holdout=best_holdout/total_iterations
    print("SNN AVG T,V,H:",avg_best_train,avg_best_val,avg_best_holdout)
    overall_siamese.append([choose,avg_best_train,avg_best_val,avg_best_holdout])
    avg_reg_acc_score=total_reg_acc_score/total_iterations
    avg_enc_acc_score=total_enc_acc_score/total_iterations
    print("TESTING AVG REG,ENC:",avg_reg_acc_score,avg_enc_acc_score)
    overall_testing.append([choose,avg_reg_acc_score,avg_enc_acc_score])

    np.savetxt("snn_network_accuracies"+str(choose)+"_s3.csv",overall_siamese[counter],delimiter=",")
    np.savetxt("snn_testing_accuracies"+str(choose)+"_s3.csv",overall_testing[counter],delimiter=",")
    counter=counter+1
print("#########################################")
print("FINAL SIAMESE ACCURACIES RESULTS:",overall_siamese)
print("FINAL TESTING ACCURACIES RESULTS:",overall_testing)
print("Total Time Elapsed: ", round(time.time()-start_time,1), "seconds")


np.savetxt("snn_network_accuracies_s3.csv",overall_siamese,delimiter=",")
np.savetxt("snn_testing_accuracies_s3.csv",overall_testing,delimiter=",")
