import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# List our data sets
from subprocess import check_output
print(check_output(["ls", "data"]).decode("utf8"))

# Loading data with Pandas
train = pd.read_csv('data/train.csv')
train_images = train.ix[:,1:].values.astype('float32')
train_labels = train.ix[:,0].values.astype('int32')

test_images = pd.read_csv('data/test.csv').values.astype('float32')

print(train_images.shape, train_labels.shape, test_images.shape)

#Show samples from training data
show_images = train_images.reshape(train_images.shape[0], 28, 28)
n = 3
row = 3
begin = 42
for i in range(begin,begin+n):
    plt.subplot(n//row, row, i%row+1)
    plt.imshow(show_images[i], cmap=plt.get_cmap('gray'))
    plt.title(train_labels[i])
    
# Normalize pixel values from [0, 255] to [0, 1]
train_images = train_images / 255
test_images = test_images / 255

# Convert labels from [0, 9] to one-hot representation.
from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)

print(train_labels[0])
print(train_images.shape, train_labels.shape)

# Create a basic neural network
# 64 relu -> 128 relu -> dropout 0.15
# -> 64 relu -> dropout 0.15 -> softmax 10 
from keras.models import Sequential
from keras.layers import Dense , Dropout

model=Sequential()
model.add(Dense(64,activation='relu',input_dim=(28 * 28)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10,activation='softmax'))

from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


# Train our model with 15 steps using 90% for training and 10% for cross validation
history=model.fit(train_images, train_labels, validation_split = 0.1, epochs=15, batch_size=64)

# Graphing Loss on the left and Accuracy on the right
history_dict = history.history

epochs = range(1, 16)

plt.rcParams["figure.figsize"] = [10,5]
plt.subplot(121)

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(epochs, loss_values, 'bo')
plt.plot(epochs, val_loss_values, 'ro')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(122)

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'ro')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show() 


# Generate prediction for test set
predictions = model.predict_classes(test_images, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("predictions.csv", index=False, header=True)
