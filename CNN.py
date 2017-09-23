from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.optimizers import SGD

# Create data generators that feed images of developing
# embryos to the CNN. Images were manually labeled
# by the competition entrant. Aggressive data augmentation
# was performed to help the CNN generalize for new cases.
train_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
  '/Users/[redacted]/Desktop/train',
  target_size=(299, 299),
  batch_size=10)

validation_generator = test_datagen.flow_from_directory(
  '/Users/[redacted]/Desktop/test',
  target_size=(299, 299),
  batch_size=10)

# Transfer learning and fine tuning functions are 
# The first removes the top layer from the existing
# Inception v3 model and adds a new softmax layer
# for the 8 classes of embryo stages used in this study.
def new_top_layer(base_model, n_classes):
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # Add a softmax layer for 2 classes
    predictions = Dense(n_classes, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# This function freezes the wights of all
# the layers in the Inception v3 model. This is the
# transfer learning aspect of training, when the
# model is fit but only allowed to adjust the
# newly initialized weights in the softmax layer.
def TL_setup(base_model, model):
    # first: train only the top layers 
    # which were randomly initialized
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    # compile the model (should be done after setting 
    # layers to non-trainable)
    model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# This function freezes the botton 290 layers only. This
# is the fine tuning aspect of training. Tne model is
# now allowd to make adjustments to the Inception v3 models
# wights in the convolution section of the model.
def FT_setup():
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:290]:
        layer.trainable = False
    for layer in model.layers[290:]:
        layer.trainable = True
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

# Instantiate the Inception v3 model
# create the base pre-trained Inception v3 model
base_model = InceptionV3(weights='imagenet', include_top=False)
# instantiate model with new top layers for embryo classes
model = new_top_layer(base_model, 2)
# setup model for transfer learning by freezing 
# all layers in the base Inception v3 model
TL_setup(base_model, model)

# Transfer learning
# Uptdate weights on top layer only
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=2,
    epochs=10,
    verbose=1,
    validation_data=validation_generator,
    nb_val_samples=2)

# Fine tuning
# Fine-tune model (fit model again but with 
# bottom 249 layers frozen)
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=2,
    epochs=10,
    verbose=1,
    validation_data=validation_generator,
    nb_val_samples=2)