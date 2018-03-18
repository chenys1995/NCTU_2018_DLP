import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Numpy XOR MLP Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
args = parser.parse_args()
np.random.seed(0)
np.set_printoptions(suppress=True, precision=4)
def sigmoid(x):
    x = 1.0/(1.0 + np.exp(-x))
    return x
# Initial weight, bias
x = np.array([[0,0],[0,1],[1,0],[1,1]])
w1 = 0.01 * np.random.randn(2,2)
b1 = np.zeros(2)
w2 = 0.01 * np.random.randn(2,1)
b2 = np.zeros(1)
y = np.array([0,1,1,0])
for i in range(args.epochs):
    # Generate permutation to x, y
    p = np.random.permutation(len(x))
    x_shuffle = np.array([x[p[i]] for i in range(len(p))])
    y_shuffle = np.array([y[p[i]] for i in range(len(p))])
    # Forward propagation
    layer1 =np.array( [np.array([
            np.array([inp[0]*w1[0][0]+inp[1]*w1[1][0]+b1[0],np.array(inp[0]*w1[0][1])+inp[1]*w1[1][1]]+b1[1])
            ]) for inp in x_shuffle]).reshape(4,2)
    sigmoid_layer1 = sigmoid(layer1)
    layer2 = np.asarray( [act[0]*w2[0]+act[1]*w2[1]+b2[0] for act in sigmoid_layer1] ).reshape(4,)
    sigmoid_layer2 = sigmoid(layer2)
    # Backward propagation
    loss = sigmoid_layer2 - y_shuffle
    print('input: ',x_shuffle.tolist())
    #print('output: ', sigmoid_layer2)
    print('loss: ' ,loss)
    #print('pred: {}'.format([(x[i].tolist(),sigmoid_layer2.reshape(4,-1)[i].tolist()) for i in range(4)]))
    #print('loss: {}'.format([(x[i].tolist(),loss.reshape(4,-1)[i].tolist()) for i in range(4)]))
    # Caculate gradient
    dsigmoid2 =  np.multiply(sigmoid_layer2, (1.0 - sigmoid_layer2), loss)
    dl2_x, dl2_w, dl2_b = dsigmoid2.reshape(4,1).dot(w2.T), sigmoid_layer1.T.dot(dsigmoid2), np.sum(dsigmoid2, axis=0)
    dsigmoid1 =  np.multiply(sigmoid_layer1, (1.0 - sigmoid_layer1), dl2_x.reshape(4,2))
    dl1_x, dl1_w, dl1_b = dsigmoid1.dot(w1.T), x.T.dot(dsigmoid1), np.sum(dsigmoid1, axis=0)
    # Update weight
    w2 -= args.lr * dl2_w.reshape(2,1)
    w1 -= args.lr * dl1_w
    #b2 -= args.lr * dl2_b
    #b1 -= args.lr * dl1_b
layer1 =np.array( [np.array([
        np.array([inp[0]*w1[0][0]+inp[1]*w1[1][0]+b1[0],np.array(inp[0]*w1[0][1])+inp[1]*w1[1][1]]+b1[1])
        ]) for inp in x]).reshape(4,2)
sigmoid_layer1 = sigmoid(layer1)
layer2 = np.asarray( [act[0]*w2[0]+act[1]*w2[1]+b2[0] for act in sigmoid_layer1] ).reshape(4,)
sigmoid_layer2 = sigmoid(layer2)
print('test output: ', sigmoid_layer2)
