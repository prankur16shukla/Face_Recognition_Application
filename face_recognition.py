# importing libraries required
# generic code can be reused for other transfer learning techniques as well
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob

# resizing the images
Image_size = [224,224]

# paths for train and test dataset
train_path = r'C:\DataScience\Notebook\Portfolio Projects\Face Recognition DL\Dataset\Train'
test_path  = r'C:\DataScience\Notebook\Portfolio Projects\Face Recognition DL\Dataset\Test'

# adding preprocessing layer
vgg = VGG16(input_shape=Image_size + [3], weights='imagenet', include_top=False)

# to prevent training of layers
for layer in vgg.layers:
    layer.trainable = False

# reading data from different folders
folders = glob(r'C:\DataScience\Notebook\Portfolio Projects\Face Recognition DL\Dataset\Train\*')

# adding layers
x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)

# creating the model
model = Model(inputs=vgg.input, outputs=prediction)

# looking at the outline of our model
model.summary()

model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# generating image data
train_datagen = ImageDataGenerator(rescale=1./255,
shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r'C:\DataScience\Notebook\Portfolio Projects\Face Recognition DL\Dataset\Train', 
target_size=(224,224), batch_size=32, class_mode='categorical')

test_set = test_datagen.flow_from_directory(r'C:\DataScience\Notebook\Portfolio Projects\Face Recognition DL\Dataset\Test', 
target_size=(224,224), batch_size=32, class_mode='categorical')

# fitting our model with data
r = model.fit_generator(
    training_set, validation_data=test_set, epochs=5, steps_per_epoch=len(training_set), validation_steps=len(test_set)
)

# saving the model
model.save(r'C:\DataScience\Notebook\Portfolio Projects\Face Recognition DL\facerecognition_model.h5')