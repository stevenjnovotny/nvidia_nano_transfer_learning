"""
this script is for generating random samples from the 
cifar10 dataset

requires a folder named 'cifar_images/' in the working 
directory
"""


from tensorflow.keras.datasets import cifar10
from sklearn.utils import shuffle
import numpy as np
from PIL import Image

(X_train, y_train), (X_val, y_val) = cifar10.load_data()
X_val, y_val = shuffle(X_val, y_val)
print(X_val.shape)


i_rand = np.random.randint(low=0, high=len(X_val)-1, size=10)
for i in i_rand:
    file_name = './cifar_images/cifar10_%r_%r.jpg' % (i,y_val[i][0]) 
    print(file_name)
    im = Image.fromarray(X_val[i,:,:,:])
    im.save(file_name)


