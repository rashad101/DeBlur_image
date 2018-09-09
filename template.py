# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import keras
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


K.tensorflow_backend._get_available_gpus()

def CNN_Model():
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Part 2 - Fitting the CNN to the images



    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('dataset/train',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('dataset/test',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    classifier.fit_generator(training_set,
                             steps_per_epoch = 1000,
                             epochs = 20,
                             validation_data = test_set,
                             validation_steps = 2000)

    return classifier

# Part 3 - Making new predictions

model = CNN_Model()

target_dir = 'models/'
if not os.path.exists(target_dir):
  print("does not exist")
  os.mkdir(target_dir)
model.save('models/model.h5')
model.save_weights('models/weights.h5')



test_image = image.load_img('dataset/single_prediction/13.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)

prediction = ""

if result[0][0] == 1:
    prediction = 'car'
else:
    prediction = 'not-car'

print(prediction)