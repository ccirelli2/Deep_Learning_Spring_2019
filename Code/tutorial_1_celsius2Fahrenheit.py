'''
Purpose:  create a model that can give the temperature in Fahrenheit when given the degrees in Celsius

'''
# Import Packages
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

# Dataset
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)


def ex_conversion(celsius_q, fahrenheit_a):
    for i, c in enumerate(celsius_q):
        print('{} degrees Celsius = {} degrees Fahrenheit'\
        .format(c, fahrenheit_a[i]))



### Create Model - Simplest is "Dense Network"------------------------------------------------

# Step1:    Create Layer(s)
'''
l0              layer 1
input_shape     [1] specifies that the input to this layer is a single value, that is, the shape 
                is a one dimensional array. 
units           specifies the number of neurons in the layer. 
'''
l0 = tf.keras.layers.Dense(units = 1, input_shape = [1])


# Step2:    Assemble Layer(s) Into a Model
'''Once layers are defined, you need to assemble them into a model'''
model = tf.keras.Sequential([l0])


# Step3:  Compile Model (Add Loss & Optomization Functions)
'''Optimizer        Here we use Adam.  The value in the parenthesis is the learning rate. 
                    This is the step sie taken when adjusting values in the model.  
                    The optimizer function here is Stochastic Gradient decent and steps 
                    are toward the local or global minimum'''
model.compile(loss='mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))


#Step4: Training the Model
'''
Weights             Initially set randomly, so the modell will need to iteratively adjust
                    to decrease the loss function. 
fit                 controls the fitting proess. 
arg1                inputs
arg2                desired output
arg3                epochs = number of times process should be conducted. 
arg4                verbose = controls how much output the model produces

'''
m_train = model.fit(celsius_q, fahrenheit_a, epochs = 500, verbose = False)
print('Finished training the model')




### Display Training Statistics---------------------------------------------------

# Import Packages
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(m_train.history['loss'])
plt.show()


















