import numpy as np
from PIL import Image
import random

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})



def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# Feed forward
def train():
    #Initializing weights
    global hiddenWeights
    hiddenWeights = np.random.rand(100, 400) * 0.000001

    global finalWeights
    finalWeights = np.random.rand(3, 100) * 0.000001

    #Initializing biases
    global hiddenBiases
    hiddenBiases = np.random.rand(100, 1) * 0.000001

    global finalBiases
    finalBiases = np.random.rand(3, 1)  * 0.000001

    for i in range(25):
        print(i)
        #Format inputs
        randomChoice = random.randint(0,2)
        im = Image.open(str(randomChoice) + '.png', 'r')
        inputs = np.reshape(np.asarray([float(i)/255.0 for i in list(im.getdata())]), (400,1))

        #Format outputs
        outputs = [0,0,0]
        outputs[randomChoice] = 1
        outputs = np.reshape(np.asarray(outputs), (3,1))

        #Calculate hidden layer
        hiddenZ = np.dot(hiddenWeights, inputs) + hiddenBiases
        hiddenLayer = sigmoid(hiddenZ)
        #print(hiddenLayer)

        #Calculate final layer
        finalZ = np.dot(finalWeights, hiddenLayer) + finalBiases
        finalLayer = sigmoid(finalZ)

        #Calculate errors
        finalError = -np.multiply(2 * (finalLayer - outputs), dsigmoid(finalZ))
        hiddenError = -np.multiply(np.matmul(finalWeights.transpose(), finalError), dsigmoid(hiddenZ))

        #Calculate changes
        changeFinalWeights = np.dot(finalError, hiddenLayer.transpose())
        changeFinalBias = finalError
        changeHiddenWeights = np.dot(hiddenError, inputs.transpose())
        changeHiddenBias = hiddenError

        #Apply changes
        finalWeights += changeFinalWeights * 0.1
        finalBiases += changeFinalBias * 0.1
        hiddenWeights += changeHiddenWeights * 0.1
        hiddenBiases += changeHiddenBias * 0.1

        # print("Final error: " + str(finalError))
        # print("Outputs: " + str(outputs))
        # print("Final layer: " + str(finalLayer))

    # print("Inputs: " + str(inputs))
    # print("Desired output: " + str(outputs))
    # print("Final layer: " + str(finalLayer))
    # print("Hidden weights: " + str(hiddenWeights))
    # print("Hidden biases: "+ str(hiddenBiases))
    # print("Final weights: " + str(finalWeights))
    # print("Final biases: "+ str(finalBiases))
    # print("Final error: " + str(finalError))

def guess():
    im = Image.open('guess.png', 'r')
    inputs = np.reshape(np.asarray([float(i)/255.0 for i in list(im.getdata())]), (400,1))

    hiddenZ = np.dot(hiddenWeights, inputs) + hiddenBiases
    hiddenLayer = sigmoid(hiddenZ)

    finalZ = np.dot(finalWeights, hiddenLayer) + finalBiases
    finalLayer = sigmoid(finalZ)

    print(finalLayer)

train()
guess()
