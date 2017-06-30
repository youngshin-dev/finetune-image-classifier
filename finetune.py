'''

You can use Inception V3 model to make your own image classifier with your own data.

Find the inception-v3.py and imagenet_utils.py here:  https://github.com/fchollet/deep-learning-models

To make the training of new layers fast, save the output of the Inception model and use it to train the new layers.

I used it to classify  images into 3 classes.

For training I used 1000 images for each class.

Start with Inception V3 network, not including last fully connected layers.

Train a simple fully connected layer on top of these.
'''

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
import inception_v3 as inception
from keras.utils import np_utils

N_CLASSES = 3
IMSIZE = (299, 299)

# TO DO:: Replace these with paths to YOUR data.
# Training directory containing separate directories for each class
train_dir = '.../train'

# Testing directory containing separate directories for each class
test_dir = '.../validation'



# Start with an Inception V3 model, not including the final softmax layer.
base_model = inception.InceptionV3(include_top=False,weights='imagenet')
print 'Loaded Inception model'

# Turn off training on base model layers
for layer in base_model.layers:
    layer.trainable = False

#test_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,  # this is the target directory
        target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=32,
        class_mode=None,
        shuffle=False)

# Data generators for feeding training/testing images to the model.
#train_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=IMSIZE,  # all images will be resized to 299x299 Inception V3 input
        batch_size=32,
        class_mode=None,
        shuffle=False)

# Run the training data on the Inception model
basemodel_output_train=base_model.predict_generator(train_generator,3000)
# Save the output
np.save(open('basemodel_output_train.npy','w'),basemodel_output_train)


# Run the test data on the Inception model
basemodel_output_test=base_model.predict_generator(test_generator,3000)
# Save the output
np.save(open('basemodel_output_test.npy','w'),basemodel_output_test)



def train_my_layers():
        train_data=np.load(open('basemodel_output_train.npy'))
        test_data=np.load(open('basemodel_output_test.npy'))
        train_labels=np.array([0]*1000+[1]*1000+[2]*1000)
        test_labels=np.array([0]*1000+[1]*1000+[2]*1000)
        #train_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        #test_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        dummy_train = np_utils.to_categorical(train_labels)
        dummy_test = np_utils.to_categorical(test_labels)

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(32,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(N_CLASSES,activation='softmax',name='my_output'))

        model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

        print (model.summary())

        print 'Trainable weights'
        print model.trainable_weights

        model.fit(train_data, dummy_train, nb_epoch=10, batch_size=32, verbose=2,validation_data=(test_data,dummy_test))

        # Save the model and trained weights for future use.
        model.save_weights('my_trained_weights2.h5')
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
                json_file.write(model_json)



train_my_layers()

