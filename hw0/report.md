# 2018 NCTU Deep Learning and Pratice Homework 0
# 1. Introduction
* This is multi layer perceptron implementation to solve XOR problem.
* Using python3.6.4 Anaconda.
* Requirement: numpy 1.14, argparse
* code: [github](https://github.com/chenys1995/NCTU_2018_DLP/tree/master/hw0)
# 2. Experiment setups
## A.Sigmoid functions
```
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
```
## B.Neural network
### Initialization
```
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0,1,1,0]]).T

num_data, input_dim = X.shape
hidden_dim = 2
output_dim = Y.T.shape[0]
W1 = np.random.random((input_dim, hidden_dim)) # hidden layer
W2 = np.random.random((hidden_dim, output_dim)) # output layer
# no bias
num_epochs = args.epochs
learning_rate = args.lr
```
### Architecture
```
layer1 = sigmoid(np.dot(layer0, W1)) 
layer2 = sigmoid(np.dot(layer1, W2)) 
```
## C.backpropagation
### Caculate gradient
```
layer2_error = Y - layer2 # calculte error
layer2_delta = layer2_error * sigmoid_backward(layer2) # 
layer1_error = np.dot(layer2_delta, W2.T)
layer1_delta = layer1_error * sigmoid_backward(layer1)
```
### Update weight
```
W2 +=  learning_rate * np.dot(layer1.T, layer2_delta)
W1 +=  learning_rate * np.dot(layer0.T, layer1_delta)
```
# 3. Results
## Hyperparameter
* Epochs: 2000
* learning rate: 0.75
## 1 hidden layer
![](https://i.imgur.com/3oPMrIY.png)
## 2 hidden layer
![](https://i.imgur.com/MDVnvXA.png)
# 4. Discussion
* backpropagation寫完，很難知道對不對。
* 剛開始寫收斂在0.5，loss -> 0。


