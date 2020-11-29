"""
Image classifier NN
nym 2020
"""

import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization, Input
class image_classifier():
    """ Class defining image properties """
    def __init__(self, im_shape):
       self.im_shape = im_shape
    
    def model_setup(self):
        """ Set up the model: Number of layers, activations, etc... """
        model = Sequential()
        model.add(Conv2D(32,(3,3),activation='relu',input_shape=self.im_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(1,activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer='adam',metrics=['accuracy'])
        self.model = model

    def train(self,X_train,Y_train):
        self.model.fit(X_train,Y_train,batch_size=32,epochs=20,verbose=2)

    def evaluate_model(self,X_test,Y_test):
        pred = self.model.evaluate(X_test,Y_test)
        print('========================================')
        print("Loss = {}".format(pred[0]))
        print("Test accuracy = {}".format(pred[1]))
    
    def predict(self,input):
        return self.model.predict(input)

def load_dataset(dataset):
    """ Prepare test data """
    dt = np.load(dataset)
    return dt

if __name__ == "__main__":
    data_location = "datasets/training_cat_dogs.h5_1000.npz"
    dt = load_dataset(data_location)
    data_location = "datasets/training_cat_dogs.h5_25000.npz"
    dt2 = load_dataset(data_location)
    X_train = np.concatenate([dt['input'][0:500,:],dt2['input'][0:500,:]])
    Y_train = np.concatenate([dt['output'][0:500],dt2['output'][0:500]])
    X_test  = np.concatenate([dt['input'][501:,:],dt2['input'][501:,:]])
    Y_test  = np.concatenate([dt['output'][501:],dt2['output'][501:]])
    im = image_classifier(X_train.shape[1:])

    X_train = X_train/255.
    X_test = X_test/255.   

    Y_train = Y_train.T
    Y_test = Y_test.T

    im.model_setup()
    im.model.summary()
    im.train(X_train,Y_train)
    im.evaluate_model(X_test,Y_test)
    a = None