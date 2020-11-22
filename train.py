"""
Implementation  of various machine learning models
nym 2020
"""

import argparse
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

class image_classifier():
    """ Class defining image properties """
    def __init__(self, width, height, n_channels):
        self.im_width  = width
        self.im_height = height
        self.im_channels = n_channels

    def training_prep(self, data_location):
        """ Training preparation """
        dog_or_cat = []
        tr_files = os.listdir(data_location)
        for t in tr_files:
            if t.split('.')[0] == 'dog':
                dog_or_cat.append(1)
            else:
                dog_or_cat.append(0)
        self.df = pd.DataFrame({
            'filename':tr_files,
            'category':dog_or_cat
         })
    
    def model_setup(self):
        """ Set up the model: Number of layers, activations, etc... """
        model = Sequential()
        model.add(Conv2D(32,(3,3),activation='relu',input_shape=(self.im_width,self.im_height,self.im_channels)))
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
        
        model.add(Dense(2,activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',metrics=['accuracy'])
        self.model = model
    
    def summary(self):
        self.model.summary()
    
    def training(self, data_location, batchsz):
        """ Train the model """
        earlystop = EarlyStopping(patience = 10)
        learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)
        callbacks = [earlystop,learning_rate_reduction]

        self.df["category"] = self.df["category"].replace({0:'cat',1:'dog'})

        train_df,validate_df = train_test_split(self.df,test_size=0.20, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)
        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]
        batch_size = batchsz

        train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                 )
        Image_Size = (self.im_width,self.im_height)
                
        train_generator = train_datagen.flow_from_dataframe(train_df,
                                                 data_location,x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
                                    
        validation_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = validation_datagen.flow_from_dataframe(
            validate_df, 
            data_location, 
            x_col='filename',
            y_col='category',
            target_size=Image_Size,
            class_mode='categorical',
            batch_size=batch_size
        )

        epochs = 10
        history = self.model.fit_generator(
            train_generator, 
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=total_validate//batch_size,
            steps_per_epoch=total_train//batch_size,
            callbacks=callbacks
        )

        self.model.save("weights/model_test_cat_or_dog_10_epochs.h5")

    def predict(self,data_location,batch_size):
        """ Predict outcome of network """
        test_df = test_data_prep(data_location)
        Image_Size = (self.im_width,self.im_height)

        train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                 )

        test_datagen = ImageDataGenerator(rotation_range=15,
                        rescale=1./255,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        width_shift_range=0.1,
                        height_shift_range=0.1)

        train_df,validate_df = train_test_split(self.df,test_size=0.20, random_state=42)
        train_df = train_df.reset_index(drop=True)

        train_generator = train_datagen.flow_from_dataframe(train_df,
                                                 data_location,x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
        test_generator = train_datagen.flow_from_dataframe(train_df,
                                                 data_location,x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
        num_samples = test_df.shape[0]
        predict = self.model.predict_generator(test_generator, steps=np.ceil(num_samples/batch_size))

        test_df['category'] = np.argmax(predict, axis=-1)
        label_map = dict((v,k) for k,v in train_generator.class_indices.items())
        test_df['category'] = test_df['category'].replace(label_map)
        test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

        sample_test = test_df.head(18)
        sample_test.head()
        plt.figure(figsize=(12, 24))
        for index, row in sample_test.iterrows():
            filename = row['filename']
            category = row['category']
            img = load_img("./dogs-vs-cats/test1/"+filename, target_size=Image_Size)
            plt.subplot(6, 3, index+1)
            plt.imshow(img)
            plt.xlabel(filename + '(' + "{}".format(category) + ')' )
        plt.tight_layout()
        plt.show()

def test_data_prep(data_location):
    """ Prepare test data """
    tests = os.listdir(data_location)
    tests_df = pd.DataFrame({
        'filename': tests
    })
    return tests_df

if __name__ == "__main__":
    im = image_classifier(128, 128, 3)
    im.training_prep("data")
    im.model_setup()
    im.summary()