# nvidia_nano_transfer_learning
This code explores transfer learning on the nano. The code nano_transfer_learning_example.py is used for creating the new model. This was done by freezing the weights in an inception_v3 model and adding layers to accomodate the cifar10 dataset.

The model was saved and run with the load_run_model.py file on the nano.

[//]: # (Image References)
[image1]: ./cifar_results.png "model"

The following shows the results on random images from the cifar10 validation set.

![alt text][image1]


The file load_sample_cifar_images.py was used to generate the random sample of images.

# Notes:

In order to get the nano_transfer_learning_example.py to run, I needed to run the following to avoid a "cannot allocate memory in the TLS block" error:

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

Also, trying to train on the nano led to a "Resource Exhausted" error. Therefore, model training was done on a macbook pro and loading/running was done on the nano.

