#!/usr/bin/env python
from model import Model
import numpy as np
import pickle as pkl
import tensorflow as tf
import sys

def to_list(prediction, length):
    list_location = [[], [], [], [], [], [], [], [], []]
    current_line = 0
    prediction = np.argmax(prediction, 2)
    print prediction.shape
    print length.shape 

    for i in range(len(prediction)):
        for j in range(length[i]):
            current_line += 1
            if prediction[i, j] == 0:
                list_location[0].append(current_line)

            elif prediction[i, j] == 1:
                list_location[1].append(current_line)

            elif prediction[i, j] == 2:
                list_location[2].append(current_line)

            elif prediction[i, j] == 3:
                list_location[3].append(current_line)

            elif prediction[i, j] == 4:
                list_location[4].append(current_line)

            elif prediction[i, j] == 5:
                list_location[5].append(current_line)

            elif prediction[i, j] == 6:
                list_location[6].append(current_line)

            elif prediction[i, j] == 7:
                list_location[7].append(current_line)

            elif prediction[i, j] == 8:
                list_location[8].append(current_line)

        current_line += 1

    return list_location

model = Model()

file = sys.argv[0]
inp = pkl.load(file)
pred, length = sess.run([model.prediction, model.length], {model.input_data: inp})

print to_list(pred, length)