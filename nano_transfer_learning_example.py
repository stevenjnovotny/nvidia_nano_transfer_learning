from tensorflow.keras.layers import Input, Lambda
import tensorflow as tf

# Loads in InceptionV3  -- could instead use resnet, vgg, etc.
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.resnet50 import ResNet50
#from keras.applications.vgg19 import VGG19

# Imports fully-connected "Dense" layers & Global Average Pooling
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.models import Model

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import cifar10

# Use a generator to pre-process our images for ImageNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
#from keras.applications.vgg19 import preprocess_input

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

"""
Example using Inception V3, although you can use the same techniques with any of the models 
in Keras Applications. 

Settting Inception to use an input_shape of 139x139x3 instead of the default 299x299x3. This 
will help speed up training. In order to do so, must set include_top to False, which means 
the final fully-connected layer with 1,000 nodes for each ImageNet class is dropped, 
as well as a Global Average Pooling layer.

Drop layers from a model with model.layers.pop(). Before, check out what the actual layers of 
the model are with Keras's .summary() function.

Notes:
If a "...TLS block" error occurs, run the following
$ export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

"""

# Set a couple flags for training
freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None
preprocess_flag = True # Should be true for ImageNet pre-trained typically
save_path = 'transfer_model'

# Use smaller than the default 299x299x3 input for InceptionV3
# which will speed up training. Keras v2.0.9 supports down to 139x139x3
input_size = 139
print(input_size)

# Using Inception with ImageNet pre-trained weights
inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=(input_size, input_size, 3))

# inception = ResNet50(weights=weights_flag, include_top=False,
#                         input_shape=(input_size, input_size, 3))

# inception = VGG19(weights=weights_flag, include_top=False,
#                         input_shape=(input_size, input_size, 3))

if freeze_flag == True:
    ## Iterate through the layers of the Inception model
    ## loaded above and set all of them to have trainable = False
    for layer in inception.layers:
        layer.trainable = False

## Use the model summary function to see all layers in the
## loaded Inception model
## * note: last two layers (global average pooling layer, 
##   and a fully-connected "Dense" layer) missing since we set include_top=false
inception.summary()

"""
Adding new layers
Insetad of using Keras's Sequential model, use the Model API. Instead of using model.add(), 
explicitly tell the model which previous layer to attach to the current layer. 

For example, if you had a previous layer named inp:

x = Dropout(0.2)(inp)
to attach a new dropout layer x, with it's input coming from a layer with the variable name inp.

Here use the CIFAR-10 dataset, which consists of 60,000 32x32 images of 10 classes. 
- airplane										
- automobile										
- bird										
- cat										
- deer										
- dog										
- frog										
- horse										
- ship										
- truck

* need to use Keras's Input function
* re-size the images up to the input_size specified earlier (139x139).
"""


# Makes the input placeholder layer 32x32x3 for CIFAR-10
cifar_input = Input(shape=(32,32,3))

# Re-sizes the input with Kera's Lambda layer & attach to cifar_input
resized_input = Lambda(lambda image: tf.image.resize( 
    image, (input_size, input_size)))(cifar_input)

# Feeds the re-sized input into Inception model
inp = inception(resized_input)



## Setting `include_top` to False earlier also removed the
## GlobalAveragePooling2D layer, but it is still needed.
x = GlobalAveragePooling2D()(inp)

## Create two new fully-connected layers using the Model API
## format. The first layer should use `out`
## as its input, along with ReLU activation. You can choose
## how many nodes it has, although 512 or less is a good idea.
## The second layer should take this first layer as input, and
## be named "predictions", with Softmax activation and 
## 10 nodes, as we'll be using the CIFAR10 dataset.
x = Dense(512, activation = 'relu')(x)
predictions = Dense(10, activation = 'softmax')(x)


"""
use model api to crate full model
"""

# Creates the model, setting final layer name to "predictions"
model = Model(inputs=cifar_input, outputs=predictions)

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check the summary of this new model to confirm the architecture
model.summary()
model.save(save_path)

"""
prep data 
"""

(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y_one_hot_train = label_binarizer.fit_transform(y_train)
y_one_hot_val = label_binarizer.fit_transform(y_val)

# Shuffle the training & validation data
X_train, y_one_hot_train = shuffle(X_train, y_one_hot_train)
X_val, y_one_hot_val = shuffle(X_val, y_one_hot_val)

# We are only going to use the first 10,000 images for speed reasons
# And only the first 2,000 images from the validation set
X_train = X_train[:10000]
y_one_hot_train = y_one_hot_train[:10000]
X_val = X_val[:2000]
y_one_hot_val = y_one_hot_val[:2000]

# Use a keras.applications generator to pre-process our images for ImageNet

if preprocess_flag == True:
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

"""
Train the model
"""

stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.0003, patience=5)
checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)

batch_size = 32
epochs = 10
# no callbacks since only are using 5 epochs 
model.fit_generator(datagen.flow(X_train, y_one_hot_train, batch_size=batch_size), \
                    steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, \
                    validation_data=val_datagen.flow(X_val, y_one_hot_val, batch_size=batch_size), \
                    validation_steps=len(X_val)/batch_size, \
                    callbacks=[stopper])

model.save(save_path)
"""
Conclusions: 
* CIFAR-10 is a fairly tough dataset 
* Should see ~70% validation accuracy
"""
