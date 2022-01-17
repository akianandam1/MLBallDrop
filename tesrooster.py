import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from BallDrop import BallDrop
import matplotlib.pyplot as plt
import random

def drop(time, initial_velocity, initial_height):
    return -4.9 * time ** 2 + initial_velocity * time + initial_height

def time_taken(initial_velocity, initial_height):
    return (-initial_velocity - np.sqrt(initial_velocity ** 2 + 4 * 4.9 * initial_height)) / (2 * -4.9)


def generate_input_point():
    initial_velocity = random.uniform(-100,100)
    initial_height = random.uniform(0,100)
    time_taken = (-initial_velocity - np.sqrt(initial_velocity ** 2 + 4 * 4.9 * initial_height)) / (2 * -4.9)
    time = random.uniform(0,time_taken)
    return np.array([time,initial_velocity,initial_height], dtype = 'float32')

#takes torch tensor as input
def vector_drop(input_vector):
    time = input_vector[0]
    initial_velocity = input_vector[1]
    initial_height = input_vector[2]
    return drop(time,initial_velocity,initial_height)

## Generates 51 random input points. Returns np array
def generate_input_set():
    result = []
    i=0
    while i <= 50:
        result.append(generate_input_point())
        i+=1
    return np.array(result, dtype='float32')

## Generates target data set based on n by 3d input set whose vectors are of of form
## [time,initial_velocity,initial_height]. Returns np array

def generate_target_set(input_set):
    result = []
    i = 0
    while i < input_set.shape[0]:
        result.append(vector_drop(input_set[i]))
        i+=1
    return np.array([result],dtype='float32')



model = BallDrop(3,500,1)

def fit(epochs, lr, model, opt_func=torch.optim.SGD):

    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        inputs = torch.from_numpy(generate_input_set())
        targets = torch.from_numpy(generate_target_set(inputs)).reshape(51, 1)

        # Training Phase

        loss = model.training_step(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # # Validation phase
        # result = evaluate(model, val_loader)
        # model.epoch_end(epoch, result)

    # return history

fit(100000, 1e-5, model)

torch.save(model.state_dict(), 'ball_model.pth')

v = 3
h = 30
t = np.linspace(0, time_taken(v,h), 1000)
y_values = []
predictions = []
for time in t:
    y_values.append(drop(time,v,h))
    predictions.append(model(torch.tensor([time,v,h],dtype=torch.float32)).item())
y_values = np.array(y_values, dtype='float32')
predictions = np.array(predictions, dtype='float32')
plt.plot(t,y_values)
plt.plot(t,predictions)
plt.show()