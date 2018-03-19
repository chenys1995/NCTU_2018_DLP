import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Numpy XOR MLP Example')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs to train (default: 5000)')
parser.add_argument('--lr', type=float, default=0.75, metavar='LR',
                    help='learning rate (default: 0.75)')
args = parser.parse_args()
np.random.seed(0)
np.set_printoptions(suppress=True, precision=4)
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x):
    return x * (1 - x)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0,1,1,0]]).T

num_data, input_dim = X.shape
hidden_dim = 2
hidden_dim2 = 8
W1 = np.random.random((input_dim, hidden_dim))
W1_2 = np.random.random((hidden_dim, hidden_dim2))
output_dim = Y.T.shape[0]
W2 = np.random.random((hidden_dim2, output_dim))
num_epochs = args.epochs
learning_rate = args.lr

for epoch_n in range(num_epochs):
    layer0 = X
    # Forward propagation.
    layer1 = sigmoid(np.dot(layer0, W1))
    layer1_2 = sigmoid(np.dot(layer1, W1_2))
    layer2 = sigmoid(np.dot(layer1_2, W2))
    # Back propagation 
    layer2_error = Y - layer2
    layer2_delta = layer2_error * sigmoid_backward(layer2)
    layer1_2_error = np.dot(layer2_delta, W2.T)
    layer1_2_delta = layer1_2_error * sigmoid_backward(layer1_2)
    layer1_error = np.dot(layer1_2_delta, W1_2.T)
    layer1_delta = layer1_error * sigmoid_backward(layer1)
    
    # update weights
    W2 +=  learning_rate * np.dot(layer1_2.T, layer2_delta)
    W1_2 +=  learning_rate * np.dot(layer1.T, layer1_2_delta)
    W1 +=  learning_rate * np.dot(layer0.T, layer1_delta)
for k,v in enumerate(X):
    layer1_act = sigmoid(np.dot(W1.T, v))
    layer1_2_act = sigmoid(np.dot(W1_2.T, layer1_act))
    prediction = sigmoid(np.dot(W2.T, layer1_2_act)) 
    print(prediction, Y[k])
