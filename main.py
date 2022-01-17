import torch
import numpy as np
from BallDrop import BallDropNet
import random


# Defines the function governing the height of the ball dropped w/ initial
# conditions after a certain time
def drop(time, initial_velocity, initial_height):
    return -4.9 * time ** 2 + initial_velocity * time + initial_height


# Returns total time the ball will spend in the air given initial conditions
def time_taken(initial_velocity, initial_height):
    return (-initial_velocity - np.sqrt(initial_velocity ** 2 + 4 * 4.9 * initial_height)) / (2 * -4.9)


# Generates a random input vector of np array form [time, initial velocity, initial height]
def generate_input_point():
    initial_velocity = random.uniform(-100, 100)
    initial_height = random.uniform(0, 100)
    time = random.uniform(0, time_taken(initial_velocity, initial_height))
    return np.array([time, initial_velocity, initial_height], dtype = 'float32')


# Takes 3d vector of form [time, initial velocity, initial height] as input
# and returns scalar representing ball's height after time elapsed.
def vector_drop(input_vector):
    time = input_vector[0]
    initial_velocity = input_vector[1]
    initial_height = input_vector[2]
    return drop(time, initial_velocity, initial_height)


# Generates input set of given size. Returns np array
def generate_input_set(size):
    result = []
    i = 0
    while i < size:
        result.append(generate_input_point())
        i += 1
    return np.array(result, dtype = 'float32')


# Generates target data set based on n by 3d input set whose vectors are of of form
# [time,initial_velocity,initial_height]. Returns np array
def generate_target_set(input_set):
    result = []
    i = 0
    while i < input_set.shape[0]:
        result.append(vector_drop(input_set[i]))
        i += 1
    return np.array([result], dtype = 'float32')


# Model takes in 3d input vector. Spits out scalar (height).
# Model has 1000 neurons per hidden layer.
model = BallDropNet(3, 1000, 1)


# Defines the function that trains the model
def fit(epochs, lr, batch_size, model, opt_func=torch.optim.SGD):

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        # Generates the input and output training data
        inputs = torch.from_numpy(generate_input_set(batch_size))
        targets = torch.from_numpy(generate_target_set(inputs)).reshape(batch_size, 1)

        # Trains the model
        loss = model.training_step(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# Run for 300,000 epochs.
fit(300000, 1e-5, 100, model)

# Saves the model
torch.save(model.state_dict(), 'FirstModel.pth')

