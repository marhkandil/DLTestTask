import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint   
from scipy.misc import imsave
import time
from keras import backend as K
from keras.constraints import non_neg


# load the pre-shuffled train and test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# rescale [0,255] --> [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# break training set into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# print shape of training set
print('x_train shape:', x_train.shape)

# print number of training, validation, and test images
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'validation samples')

#First Network: Commented it out


# model = Sequential()
# model.add(Conv2D(filters=6, kernel_size=5, strides=1, padding='same', 
#     activation='relu', input_shape=(32, 32, 3), kernel_constraint= non_neg()))
# model.add(MaxPooling2D(pool_size=2, strides=2))
# model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', 
#     activation='relu'))
# model.add(MaxPooling2D(pool_size=2, strides=2))
# model.add(Conv2D(filters=120, kernel_size=5, strides=1, padding='same', 
#     activation='relu'))
# model.add(Flatten())
# model.add(Dense(84, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.summary()

# Second Network:


model = Sequential()
model.add(Conv2D(filters=6, kernel_size=11, strides=1, padding='same', 
    activation='relu', input_shape=(32, 32, 3), kernel_constraint= non_neg()))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=16, kernel_size=11, strides=1, padding='same', 
    activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=120, kernel_size=11, strides=1, padding='same', 
    activation='relu'))
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# compile the model
adm = keras.optimizers.Adam(lr=0.0003)
model.compile(loss='categorical_crossentropy', optimizer=adm, 
                  metrics=['accuracy'])


# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=10,
          validation_data=(x_valid, y_valid), callbacks=[checkpointer], 
          verbose=2, shuffle=True)

# load the weights that yielded the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

#Code for visualizing filters

# dimensions of the generated pictures for each filter.
img_width =32
img_height =32

# the name of the layer we want to visualize 
# (see model definition at keras/applications/vgg16.py)
layer_name = 'conv2d_1'

# util function to convert a tensor into a valid image


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input
# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])

filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

# we start from a gray image with some noise
input_img_data = np.random.random((1, img_width, img_height, 3))
# run gradient ascent for 20 steps
step = 1
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step
    
img = input_img_data[0]
img = deprocess_image(img)
imsave('%s_filter_%d.png' % (layer_name, filter_index), img)

#Code for plotting out the weights of each of the filters in the first layer of the network

w2 = model.layers[0].get_weights()[0]
#View shape of the np-array
np.shape(w2)
#replace the 5 with the filterIndex
w2 = w2[:,:,:,5]

plt.imshow(w2)
plt.show()
#Code for generating the plotted graphs of accuracy and loss

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

plot_model_history(hist)
