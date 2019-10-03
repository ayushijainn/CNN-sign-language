# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
from PIL import Image

from keras.models import model_from_json

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units= 128, activation = 'relu'))
classifier.add(Dense(units= 24, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 3840,
                         nb_epoch =20,
                         validation_data = test_set,
                         nb_val_samples = 720)
history=classifier.fit_generator(training_set,
                         samples_per_epoch = 3840,
                         nb_epoch =50,
                         validation_data = test_set,
                         nb_val_samples = 720)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model_json=classifier.to_json()
with open("model.json",'w') as json_file:
    json_file.write(model_json)
    
classifier.save_weights("model.h5")
print("saved model to disk")



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('single_prediction/sign_y.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result =loaded_model.predict(test_image)


a = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
b = [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
c = [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
d = [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
e = [[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
f = [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
g = [[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
h = [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
i = [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
k = [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
l = [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
m = [[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]]
n = [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]]
o = [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]]
p = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]]
q = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]]
r = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]]
s = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]]
t = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]]
u = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]]
v = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]]
w = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]]
x = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]]
y = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]



result=result.astype (int)
print (result)
result = result.tolist()
print (result)

training_set.class_indices

if(result==a):
    label=1
    print ("the gesture predicted is A ")

elif(result==b):
    label=2
    print ("the gesture predicted is B ")

elif(result==c):
    label=3
    print ("the gesture predicted is C ")

elif (result==d):
    label=4
    print ("the gesture predicted is D ")
elif (result==e):
    label=5
    print ("the gesture predicted is E ")
elif (result==f):
    label=6
    print ("the gesture predicted is F ")
elif (result==g):
    label=7
    print ("the gesture predicted is G ")
elif (result==h):
    label=8
    print ("the gesture predicted is H ")
elif (result==i):
    label=9
    print ("the gesture predicted is I ")
elif (result==k):
    label=10
    print ("the gesture predicted is K ")
elif (result==l):
    label=11
    print ("the gesture predicted is L ")
elif (result==m):
    label=12
    print ("the gesture predicted is M ")
elif (result==n):
    label=13
    print ("the gesture predicted is N ")
elif (result==o):
    label=14
    print ("the gesture predicted is O ")
elif (result==p):
    label=15
    print ("the gesture predicted is P ")
elif (result==q):
    label=16
    print ("the gesture predicted is Q ")
elif (result==r):
    label=17
    print ("the gesture predicted is R ")
elif (result==s):
    label=18
    print ("the gesture predicted is S ")
elif (result==t):
    label=19
    print ("the gesture predicted is T ")
elif (result==u):
    label=20
    print ("the gesture predicted is U ")
elif (result==v):
    label=21
    print ("the gesture predicted is V ")
elif (result==w):
    label=22
    print ("the gesture predicted is W ")
elif (result==x):
    label=23
    print ("the gesture predicted is X ")
elif (result==y):
    label=24
    print ("the gesture predicted is Y ")
    
else:
    print("No image predicted")
    
print ("Label of the gesture is",label)
