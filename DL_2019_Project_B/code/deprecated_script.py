# GSU - DEEP LEARNING - PROJECT B
'''Readings
    https://www.tensorflow.org/guide/keras

'''

# LIBRARIES________________________________________________________________
# Load Standard Libraries
import pandas as pd
import os
import mysql.connector
from datetime import datetime

# Import Personal Modules
import module1 as m1

# Load Scikit Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load Keras Libraries
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout, Activation


# DATA_____________________________________________________________________
# Create Connection to MySQL DB
mydb = mysql.connector.connect(
        host="localhost",
        user="ccirelli2",
        passwd="Work4starr",
        database='GSU'
        )

# Clean Data
df = m1.get_labeled_data(mydb)

# Load Data
df_new = pd.read_excel('Newdataset.xlsx')
df_new['index'] = df['LABEL']


# DATA PREP____________________________________________________________

# Split X & Y
y_raw       = df['LABEL'].values
y           = pd.get_dummies(y_raw).values                 # convert y to dummy cols
sentences   = df['TEXT'].values

# Split Train & Test
sentences_train, sentences_test, y_train, y_test = train_test_split(
                    sentences, y, test_size=0.30, random_state=1000)

# Vectorize
vectorizer = CountVectorizer(min_df = 3, lowercase = True, stop_words = 'english', 
                             max_features = 10000)
vectorizer.fit(sentences_train)
x_train = vectorizer.transform(sentences_train).toarray()
x_test  = vectorizer.transform(sentences_test).toarray()


# MODEL_____________________________________________________________________

# Convert X Values Into an Array
input_dim = x_train.shape[1]            # Number of features
print('\nNumber of words => {}\n'.format(input_dim))

# Instantiate Model
model = Sequential()

# Add Layers
'''input_shape: specifies the dimension of the input to the layer
    activation:  function used to activate the layer
    units:       number of neurons in the layer.  if our output is 1, and we are on the last
                layer, then units should = 1.'''

# Build Model

model.add(layers.Dense(units = 90, input_dim = input_dim, activation = 'relu'))
model.add(Dropout(0.25))
#model.add(layers.Dense(units = 64, activation = 'relu'))
#model.add(layers.Dense(units = 32, activation = 'relu'))
#model.add(layers.Dense(units = 18))
model.add(layers.Dense(units = 9, input_dim= input_dim, activation = 'softmax'))


# Specify Optomizer
model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
# Prints Summary of NN Structure
model.summary()

# Set Number of Epochs
history = model.fit(x_train, y_train, 
                    epochs=60, 
                    verbose=False, 
                    validation_data=(x_test, y_test),
                    batch_size=500)

# Measure Accuracy of Model
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print('Training Accuracy: {}'.format(round(accuracy,4)))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print('Test Accuracy: {}'.format(round(accuracy, 2)))


# Plot 
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)
plt.show()

