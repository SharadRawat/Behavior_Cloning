
# coding: utf-8

# ### Project 3 - Behavioral Cloning
# 
# This is the 3rd project for term1 of SDCND.

# In[1]:

#import necessary libraries
import csv
import sklearn
import cv2
import tensorflow as tf
import keras
import numpy as np
import matplotlib as plt

## Reading the .csv file for data input
lines=[]
with open('../CarND-Behavioral_Cloning/driving_log.csv') as csvfile:
    read_file = csv.reader(csvfile)
    for line in read_file:
        lines.append(line)


## reading the files by changing the address from local machine to the address on the instance

images,measurements =[],[]
total_images,total_measurements =[],[]
correction_factor = 0.25                                                  # Steering correction factor (0.25 gave less going off road)
for i in range(3):                                                        # loop for left,right and centre images
    for line in lines:
            source_path = line[i]
            filename = source_path.split('\\')[-1]                         
            current_path = '../CarND-Behavioral_Cloning/IMG/' + filename
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(320, 160))
            images.append(image)
            measurement = float(line[3])

            if i == 0:
                measurements.append(measurement)
            elif i == 1:
                measurements.append(measurement + correction_factor)
            else:
                measurements.append(measurement - correction_factor)
    total_images.extend(images)
    total_measurements.extend(measurements)


images = total_images
measurements = total_measurements


# Images were augmented by flipping them about vertical axis. 
# Images can also be rotated,this will improve the model for hilly areas. 
# Not done because a large dataset was already obtained and training was time consuming.


augmented_image, augmented_measurements = [],[]                              
for image, measurement in zip(images,measurements):
    augmented_image.append(image)
    augmented_measurements.append(measurement)
    augmented_image.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    




X_train = np.array(augmented_image)
y_train = np.array(augmented_measurements)

#Keras modules imported Image was cropped 50 from top and 20 from bottom.
            

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5,input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))


'''
################## Previous half model)
model.add(Convolution2D(16, 5, 5))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))'''
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))

model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))

model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))

model.add(Convolution2D(64,3,3, activation='relu'))

model.add(Convolution2D(64,3,3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))





model.compile(loss='mse',optimizer='Adam')
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5,verbose=1)
model.save('model.h5')

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])


# In[2]:

'''
import libraries
import csv
from sklearn.utils import shuffle
import cv2
import tensorflow as tf
import keras
import numpy as np
import matplotlib as plt

lines=[]
with open('../CarND-Behavioral_Cloning/driving_log.csv') as csvfile:
    read_file = csv.reader(csvfile)
    for line in read_file:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images,measurements =[],[]
            total_images,total_measurements =[],[]
            correction_factor = 0.25
            for i in range(3):
                for line in lines:
                        source_path = line[i]
                        filename = source_path.split('\\')[-1]                         
                        current_path = '../CarND-Behavioral_Cloning/IMG/' + filename
                        image = cv2.imread(current_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image,(320, 160))
                        images.append(image)
                        measurement = float(line[3])

                        if i == 0:
                            measurements.append(measurement)
                        elif i == 1:
                            measurements.append(measurement + correction_factor)
                        else:
                            measurements.append(measurement - correction_factor)
                total_images.extend(images)
                total_measurements.extend(measurements)



            images = total_images
            measurements = total_measurements
            images,measurements =[],[]
            total_images,total_measurements =[],[]
            augmented_image, augmented_measurements = [],[]
            for image, measurement in zip(images,measurements):
                augmented_image.append(image)
                augmented_measurements.append(measurement)
                augmented_image.append(cv2.flip(image,1))
                print(cv2.imshow(augmented_image[1]))
                augmented_measurements.append(measurement*-1.0)




            X_train = np.array(augmented_image)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5,input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((75,25),(0,0))))

model.add(Convolution2D(6, 5, 5))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(6, 3, 3))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))


model.add(Flatten())

model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))

#model.add(Flatten())
#model.add(Dense(1))

model.add(Convolution2D(16, 5, 5))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1176))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(1))





model.compile(loss='mse',optimizer='Adam')
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=10,verbose=1)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('model.h5')'''


# In[ ]:



