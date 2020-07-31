import os
import time

# for macOS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
load pre-trained model
"""

from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import decode_predictions
from tensorflow.keras.applications.inception_v3 import preprocess_input

start = time.time()
model = keras.models.load_model('nano_transfer_model', custom_objects={"tf": tf, "input_size": 139})
model.summary()
print( 'model loading time: %.2f seconds\n' % (time.time() - start))

"""
test on some sample images
"""
from glob import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image_paths = glob('cifar_images/*.jpg')

# Print out the image paths
print(image_paths)

# predict and plot
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#fig, ax = plt.subplots(nrows = , ncols=5, figsize=(10,4))
nrows = int(np.ceil(len(image_paths)/5.0))
ncols = 5
plt.figure(figsize = (ncols*2, nrows*1.5))

start = time.time()
for ix, item in enumerate(image_paths):

    #example = mpimg.imread(item)
    img = image.load_img(item, target_size=(32, 32))
    x = image.img_to_array(img)
    #image = tf.image.resize(example, (32, 32))
    #image = np.expand_dims(image, axis=0)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)

    # Check the top 3 predictions of the model
    #print('Predicted:', decode_predictions(predictions, top=3)[0])

    #predictions = model.predict(image)
    print('Predicted:', predictions)

    # View an example of an image

    label = labels[np.argmax(predictions)]
    print(label, int(item[-5]))
    ax = plt.subplot(nrows, ncols, ix + 1)
    ax.imshow(x[0,:,:,:])
    ax.set_title('P: %s\nT: %s' % (label, labels[int(item[-5])]))
    ax.axis('off')



print( 'model loading time: %.2f seconds\n' % (time.time() - start))
plt.subplots_adjust(wspace=None, hspace=0.65)

plt.show(block=True)
