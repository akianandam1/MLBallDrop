import torch
from BallDrop import BallDrop
import numpy as np
import matplotlib.pyplot as plt

imported_model = BallDrop(3,500,1)
imported_model.load_state_dict(torch.load('ball_model.pth'))

def time_taken(initial_velocity, initial_height):
    return (-initial_velocity - np.sqrt(initial_velocity ** 2 + 4 * 4.9 * initial_height)) / (2 * -4.9)

def drop(time, initial_velocity, initial_height):
    return -4.9 * time ** 2 + initial_velocity * time + initial_height


def plotcurves(model, axis,v,h):
    t = np.linspace(0, time_taken(v, h), 1000)
    y_values = []
    predictions = []
    for time in t:
        y_values.append(drop(time, v, h))
        predictions.append(model(torch.tensor([time, v, h], dtype = torch.float32)).item())
    y_values = np.array(y_values, dtype = 'float32')
    predictions = np.array(predictions, dtype = 'float32')
    axis.plot(t, y_values)
    axis.plot(t, predictions)

figure, axis = plt.subplots(2, 2)
plotcurves(imported_model, axis[0,0], -20, 5)
plotcurves(imported_model, axis[1,0], 8, 51)
plotcurves(imported_model, axis[0,1], 70, 5)
plotcurves(imported_model, axis[1,1], -3, 87)


plt.show()