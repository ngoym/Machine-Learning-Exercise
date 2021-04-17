"""
Image classifier NN
nym 2020
"""

import argparse
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization, Input, Add, \
     ZeroPadding2D, GlobalMaxPooling2D,\
     AveragePooling2D

from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)    
    return X

class resnet():
    """ Resnet base class """
    def __init__(self, im_shape):
       self.im_shape = im_shape
       self.classes  = 1
    
    def model_setup(self):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        classes = self.classes
        X_input = Input(self.im_shape)
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        # Stage 1
        X = Conv2D(filters=64, kernel_size=(7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
        # Stage 2
        X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
        # Stage 3 
        X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
        X = identity_block(X, 3, [128,128,512], stage=3, block='b')
        X = identity_block(X, 3, [128,128,512], stage=3, block='c')
        X = identity_block(X, 3, [128,128,512], stage=3, block='d')
        # Stage 4
        X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
        # Stage 5
        X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        X = AveragePooling2D(pool_size=(2,2))(X)

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        # Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet50')
        model.compile(loss='binary_crossentropy',
                optimizer='adam',metrics=['accuracy'])
        self.model = model

    def train(self,X_train,Y_train):
        filepath="models/resnet-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.model.fit(X_train,Y_train,batch_size=32,epochs=30,verbose=2,callbacks=callbacks_list,validation_split=0.25)

    def evaluate_model(self,X_test,Y_test):
        pred = self.model.evaluate(X_test,Y_test)
        print('========================================')
        print("Loss = {}".format(pred[0]))
        print("Test accuracy = {}".format(pred[1]))
    
    def predict(self,input):
        return self.model.predict(input)

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
        filepath="models/imgnet-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.model.fit(X_train,Y_train,batch_size=32,epochs=30,verbose=0,callbacks=callbacks_list,validation_split=0.25)

    def evaluate_model(self,X_test,Y_test):
        pred = self.model.evaluate(X_test,Y_test)
        print('========================================')
        print("Loss = {}".format(pred[0]))
        print("Test accuracy = {}".format(pred[1]))
    
    def predict(self,input):
        return self.model.predict(input)

class Classifier():
    """ Select imagae classifier """
    def __init__(self,im_shape, choice="default"):
        if choice == "default":
            self.m = image_classifier(im_shape)
        elif choice == "resnet":
            self.m = resnet(im_shape)
    def get_model(self):
        return self.m

def load_dataset(dataset):
    """ Prepare test data """
    dt = np.load(dataset)
    return dt

if __name__ == "__main__":
    """
    data_location = "datasets/training_cat_dogs.h5_1000.npz"
    dt = load_dataset(data_location)
    data_location = "datasets/training_cat_dogs.h5_25000.npz"
    dt2 = load_dataset(data_location)
    X_train = np.concatenate([dt['input'][0:500,:],dt2['input'][0:500,:]])
    Y_train = np.concatenate([dt['output'][0:500],dt2['output'][0:500]])
    X_test  = np.concatenate([dt['input'][501:,:],dt2['input'][501:,:]])
    Y_test  = np.concatenate([dt['output'][501:],dt2['output'][501:]])
    """

    to_load = [
        'training_cat_dogs.h5_1000.npz',
        'training_cat_dogs.h5_2000.npz',
        'training_cat_dogs.h5_3000.npz',
        'training_cat_dogs.h5_25000.npz',
        'training_cat_dogs.h5_24000.npz',
        'training_cat_dogs.h5_23000.npz'
    ]

    first = True
    for k in to_load:
        data_location = f"datasets/{k}"
        dt = load_dataset(data_location)
        if first:
            first = False
            X = dt['input']
            Y = dt['output']
        else:
            X = np.concatenate([X,dt['input']])
            Y = np.concatenate([Y,dt['output']])
    n = np.random.permutation(Y.shape[0])
    X = X[n,:]
    if len(Y.shape) == 1:
        Y = Y[n]
    else:
        Y = Y[n,:]

    TEST_SIZE = 1000

    Y_test  = Y[0:TEST_SIZE]
    Y_train = Y[TEST_SIZE:]
    X_test  = X[0:TEST_SIZE,:]
    X_train = X[TEST_SIZE:,:]

    X_train = X_train/255.
    X_test = X_test/255.   

    Y_train = Y_train.T
    Y_test = Y_test.T

    #im = image_classifier(X_train.shape[1:])
    imm = Classifier(X_train.shape[1:],choice='default')
    im = imm.get_model()
    im.model_setup()
    im.model.summary()
    im.train(X_train,Y_train)
    im.evaluate_model(X_test,Y_test)
    a = None

    imm = Classifier(X_train.shape[1:],choice='resnet')
    im = imm.get_model()
    im.model_setup()
    im.train(X_train,Y_train)
    im.evaluate_model(X_test,Y_test)