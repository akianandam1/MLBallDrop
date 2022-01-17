import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

## Input in the format time, initial_velocity, initial_heigt
user_input = np.array([2, 10, 3])


## Returns the height of a ball dropped after a
## certain time w/ initial height and velocity
def drop(time, initial_velocity, initial_height):
    return -4.9 * time ** 2 + initial_velocity * time + initial_height

def time_taken(initial_velocity, initial_height):
    return (-initial_velocity - np.sqrt(initial_velocity ** 2 + 4 * 4.9 * initial_height)) / (2 * -4.9)


## Function that returns a set of time input points based on position and
## velocity of ball drop. Returns numpy array

def time_data(initial_velocity, initial_height):
    ## calculates how long it will take for the ball to drop
    time_taken = (-initial_velocity - np.sqrt(initial_velocity ** 2 + 4 * 4.9 * initial_height)) / (2 * -4.9)

    ## returns a set of 1000 equally spaced points from 0 to the time the ball hits the ground
    return np.linspace(0, time_taken, 1000)

# times = torch.from_numpy(time_data(3,30))
# time_data = torch.reshape(times, (1,times.shape[0]))
# print(time_data)
# velocities = torch.reshape(torch.from_numpy(np.full(times.shape[0],3)),(1,times.shape[0]))
# heights = torch.reshape(torch.from_numpy(np.full(times.shape[0],30)),(1,times.shape[0]))
#
# inputs = torch.cat((velocities,heights,time_data))
# print(inputs)


## Gets the full data set for ball drop w/initial velocity & height.
def get_data_set(initial_velocity, initial_height):
    # Converts numpy input data to torch tensor
    time_inputs = torch.from_numpy(time_data(initial_velocity, initial_height))
    # Gets output data
    outputs = drop(time_inputs, initial_velocity, initial_height)
    # Returns torch data set
    return TensorDataset(time_inputs, outputs)

## Gets a small testing batch of data for each
## iteration in the optimization. Automatically shuffles the set
def get_batch_data(data_set, batch_size):
    return DataLoader(data_set, batch_size, shuffle = True)

# for x,y in get_batch_data(get_data_set(3,30),20):
#     print(x)
#     print(y)
#     break

## Starting w/linear model

## The initial velocities will range from -100 and 100 for testing.
## The initial heights will range from 0 to 100 for testing.
## The time can range from anywhere from 0 to the final time depending on initial velocity.


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


model = nn.Linear(3,1)
loss_fn = F.mse_loss
# loss = loss_function(model(inputs), targets)
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

inputs = torch.from_numpy(generate_input_set())
targets = torch.from_numpy(generate_target_set(inputs)).reshape(51, 1)
# print(inputs)
# print(targets)

def fit(num_epochs, model, loss_fn, opt):
    # Repeat for given number of epochs

    inputs = torch.from_numpy(generate_input_set())
    targets = torch.from_numpy(generate_target_set(inputs)).reshape(51, 1)

    for epoch in range(num_epochs):

        i = 0
        while i <= 50:
            pred = model(inputs[i])

                # 2. Calculate loss
            loss = loss_fn(pred, targets[i])

                # 3. Compute gradients
            loss.backward()

                # 4. Update parameters using gradients
            opt.step()

                # 5. Reset the gradients to zero
            opt.zero_grad()
            i += 1

            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))



# fit(1000, model, loss_fn, opt)
#
# # print(model(torch.tensor([2,3,30], dtype=torch.float32)))
#
# v = 3
# h = 30
# t = np.linspace(0, time_taken(v,h), 1000)
# y_values = []
# predictions = []
# for time in t:
#     y_values.append(drop(time,v,h))
#     predictions.append(model(torch.tensor([time,v,h],dtype=torch.float32)).item())
# y_values = np.array(y_values, dtype='float32')
# predictions = np.array(predictions, dtype='float32')
# plt.plot(t,y_values)
# plt.plot(t,predictions)
# plt.show()

# new_model = nn.Linear(5,6)
# print(new_model.weight)
# print(new_model.bias)
# torch.save(new_model.state_dict(), 'test.pth')

# third_model = nn.Linear(5,6)
# third_model.load_state_dict(torch.load('test.pth'))
# print(third_model.weight)
# print(third_model.bias)