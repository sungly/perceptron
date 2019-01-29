import numpy as np
import csv
from random import *

# global constants 
seed_file = "trainSeeds.csv"

'''
Extract all rows from a given CSV file into an array 
'''
def extract_csv(file_name):
    inputs = []
    csv_reader = csv.reader(open(file_name), delimiter=",")

    # convert each value into float 
    for row in csv_reader:
        row = list(map(float, row))
        inputs.append(row)
    
    return inputs

class Perceptron():
    weights = None
    threshholds = [0.0, 0.0, 0.0]
    active_nodes = [False, False, False]

    def __init__(self, learning_rate=1, epoch=50):
        ''' 
        Initializes the weights to zeros.Since the length of each input 
        row also includes the expected result, the weights array length 
        will be equal tp input_row_len to accomodate for the bias value 
        at index 0. 
        '''
        # self.weights = np.zeros((input_row_len,), dtype=np.float)
        # self.row_length = input_row_len - 1
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.train()
    
    '''
    Calculates the total summation of input attributues * the weights 
    of the network. 

    z = summation(0-n) { WiXi}, where w = weight, x = input attribute 
    '''
    def input_summation(self, attributes, class_type):
        return np.dot(attributes, self.weights[class_type])

    '''
    Determines whether the node is fired, that is if the activation is 
    greater than the threshhold value for that class. 
    '''
    def activation(self, value, class_type):
        print("Value: ", value)
        print("Threshhold: ", self.threshholds[class_type])
        if value > self.threshholds[class_type]:
            return 1
        else:
            return 0

    '''
    Adjusts the weight values based on the result of the prediction.
    Also update the threshold value for each classification.
    '''
    def feedback_learning(self, data, actual_output, expected_output, class_type):
        if actual_output > expected_output:
            for i in range(len(self.weights[0])):
                self.weights[class_type][i] = self.weights[class_type][i] - (self.learning_rate * data[i])
            self.threshholds[class_type] += self.threshholds[class_type] - (self.learning_rate * data[i])
        else:
            for i in range(len(self.weights[0])):
                self.weights[class_type][i] = self.weights[class_type][i] + (self.learning_rate * data[i])
            self.threshholds[class_type] += self.threshholds[class_type] + (self.learning_rate * data[i])

        print("threshhold updated to: ", self.threshholds)
    '''
    Predict what type a given wheat is based off of its attributes 
    and the current weight values. 

    If the perceptron produces the wrong result, apply feedback learning.
    '''
    def predict(self, data, expected_output, class_type):
        print("Class: ", class_type + 1)
        summation = self.input_summation(data, class_type)
        activation_fire = self.activation(summation, class_type)
        y = 0
        d = 0

        print("activation: ", activation_fire)
        if activation_fire == 1 and expected_output == class_type + 1:
            # correct prediction 
            self.active_nodes[class_type] = True
            #message = "Correct prediction: " + 'Expected: ' + str(expected_output) + ", Actual: " + str(class_type + 1)
            # print(message)
        elif activation_fire == 1 and expected_output != class_type + 1:
            # case of y=1, d=0
            y = 1
            d = 0
            self.feedback_learning(data, y, d, class_type)
            # message = "Wrong prediction: " + 'Expected: ' + str(expected_output) + ", Actual: " + str(class_type + 1)
            # print(message)
            # print("Weight Vector changed to: ", self.weights[class_type])
            
        elif activation_fire == 0 and expected_output == class_type + 1:
            # case of y=0, d=1
            y = 0
            d = 1
            self.feedback_learning(data, y, d, class_type)
            # message = "Wrong prediction: " + 'Expected: ' + str(expected_output) + ", Actual: " + str(class_type + 1)
            # print(message)
            # print("Weight Vector changed to: ", self.weights[class_type])
        # else:
            # case of y=0, d=0
            

        

    '''
    Initializes the weight vectors for each different class
    self.weights[0] - class 1 weight vector 
    self.weights[1] - class 2 weight vector 
    self.weights[2] - class 3 weight vector 
    '''
    def init_weights(self, length):
        self.weights = [
            [0] + [randint(0, 1)  for i in range(length-1)],
            [0] + [randint(0, 1)  for i in range(length-1)],
            [0] + [randint(0, 1)  for i in range(length-1)]
        ]

    '''
    Trains the model to predict the correct type of wheat. 
    @NOTE: x0 is defined to be a constant 1.
    @TODO: add epoch 
        - reset self.active_nodes 
    '''
    def train(self):
        inputs = extract_csv(seed_file)
        
        one_row_length = len(inputs[0])
        self.init_weights(one_row_length)

        print("Training perception...")

        for _ in range(1):
            inputs = extract_csv(seed_file)
            for input_row in inputs:
                data = [1] + input_row[0:one_row_length-1]
                expected_output = input_row[-1]
            
                for i in range(3):
                    self.predict(input_row, expected_output, i)

                print("Expected OUtout: ", expected_output)
                print("Final active nodes: ", self.active_nodes)
                self.active_nodes = [False, False, False]
                
                # check which one it belongs to 
                # reset self.active_nodes here
        print("Finished training!")

p = Perceptron()