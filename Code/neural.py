import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
len(x_train)
# 60000
len(x_test)
# 10000
# Finding the shape of individual sample
x_train[0].shape
x_train[0]
plt.matshow(x_train[0])

y_train[0]
# Show first 5 data
y_train[:5]
x_train.shape

# Scale the data so that the values are from 0 - 1
x_train = x_train / 255
x_test = x_test / 255
x_train[0]
# Flattening the train and test data
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)
x_train_flattened.shape


# Sequential create a stack of layers
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
# Optimizer will help in backproagation to reach better global optima
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Does the training
model.fit(x_train_flattened, y_train, epochs=10, validation_split = 0.5)

model.evaluate(x_test_flattened, y_test)

# Show the image
plt.matshow(x_test[0])

# Make the predictions
y_predicted = model.predict(x_test_flattened)
y_predicted[0]

# Find the maximum value using numpy
np.argmax(y_predicted[0])

# In[11]:
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
cm

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Sequential create a stack of layers
# Create a hidden layer with 100 neurons and relu activation
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
# Optimizer will help in backproagation to reach better global optima
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Does the training
model.fit(x_train_flattened, y_train, epochs=25)
# In[ ]:
model.evaluate(x_test_flattened, y_test)
# In[ ]:
y_predicted = model.predict(x_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')





