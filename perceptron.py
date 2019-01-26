import numpy as np
import csv

# global constants 
seed_file = "testSeeds.csv"

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

    def __init__(self, threshhold=150, learning_rate=0.1):
        ''' 
        Initializes the weights to zeros.Since the length of each input 
        row also includes the expected result, the weights array length 
        will be equal tp input_row_len to accomodate for the bias value 
        at index 0. 
        '''
        # self.weights = np.zeros((input_row_len,), dtype=np.float)
        # self.row_length = input_row_len - 1
        self.threshhold = threshhold
        self.learning_rate = learning_rate
        self.train()
    
    '''
    Calculates the total summation of input attributues * the weights 
    of the network. 

    z = summation(0-n) { WiXi}, where w = weight, x = input attribute 
    '''
    def input_summation(self, input_row):
        return np.dot(self.weights, input_row)

    '''
    Returns the output wheat type:
    Type A - 1 
    Type B - 2
    Type C - 3 
    '''
    def activation(self, value):
        print("Summation: ", value)
        if (value <= 0.3):
            return 1
        elif (value >= 0.4 and value <= 0.7):
            return 2
        else:
            return 3

    '''
    @NOTE: the last value in input_row is the expected output

    Adjusts the weight values based on the result of the prediction.
    '''
    def feedback_learning(self, input_row, actual_output):
        expected_output = input_row[-1]

        # bias value 
        self.weights[0] += self.learning_rate

        for i in range(1, len(self.weights)):
            if(actual_output >= expected_output):
                self.weights[i] = self.weights[i] - (self.learning_rate * input_row[i-1])
            else:
                self.weights[i] = self.weights[i] + (self.learning_rate * input_row[i-1])
        
        print("Adjusted weights to: ", self.weights)

    '''
    Predict what type a given wheat is based off of its attributes 
    and the current weight values. 

    If the perceptron produces the wrong result, apply feedback learning.
    '''
    def predict(self, input_row):
        summation = self.input_summation(input_row)
        wheat_type = self.activation(summation)

        print("Expected output: ", input_row[-1])
        print("Actual output: ", wheat_type)

        if (wheat_type != input_row[-1]):
            self.feedback_learning(input_row, wheat_type)
        else:
            print("Prediction Correct!")

    '''
    Trains the model to predict the correct type of wheat. 
    @NOTE: took out training_inputs for testing
    '''
    def train(self):
        inputs = [18.98,16.57,0.8687,6.449,3.552,2.144,6.453,2]
        length = len(inputs)
        self.weights = np.zeros((length,), dtype=np.float)

        self.predict(inputs)




# a = extract_csv(seed_file)
p = Perceptron()

'''
[18.98,16.57,0.8687,6.449,3.552,2.144,6.453,2]
'''