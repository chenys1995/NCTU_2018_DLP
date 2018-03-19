import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Numpy XOR MLP Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
args = parser.parse_args()
np.random.seed(0)
np.set_printoptions(suppress=True)
def sigmoid(x):
    x = 1.0/(1.0 + np.exp(-x))
    return x
def sigmoid_backward(x):
    dx = x * (1-x)
    return dx
def affine_forward(x,w,b):
    row_dim = x.shape[0]
    col_dim = np.prod(x.shape[1:])
    x_reshape = x.reshape(row_dim, col_dim)
    out = np.dot(x_reshape, w) + b
    out = sigmoid(out)
    return out, (x,w,b)
def affine_backward(dout,cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    row_dim = x.shape[0]
    col_dim = np.prod(x.shape[1:])
    x_reshape = x.reshape(row_dim, col_dim)
    dw = x_reshape.T.dot(dout)
    dx = dout.dot(w.T).reshape(x.shape)
    db = np.sum(dout, axis=0)
    return dw, dx, db
x = np.array([[0,0],[0,1],[1,0],[1,1]])
w1 = 0.1 * np.random.randn(2,8)
b1 = np.zeros(8)
#b1 = np.array([-10,30])
w2 = 0.1 * np.random.randn(8,1)
b2 = np.zeros(1)
#b2 = np.array([-30])
y = np.array([0,1,1,0])
for i in range(args.epochs):
    #Forward 
    avg = np.zeros(1)
    mb2,mb1,ub1,ub2 = 0.0,0.0,0.0,0.0
    for k in np.random.permutation(4):
        a1out, a1cache = affine_forward(np.array([x[k]]),w1,b1)
        siga1out = sigmoid(a1out)
        a2out, a2cache = affine_forward(siga1out,w2,b2)
        siga2out = sigmoid(a2out)
        #calculate loss
        loss = a2out - y[k]
        print('data: {}, out:{}, loss: {}'.format(x[k],a2out,loss[0]))
        #print(loss)
        avg += loss[0]
        #Backward
        #a2cache = a2cache[0], a2cache[1].reshape(2,1), a2cache[2]
        dsig2 = sigmoid_backward(siga2out) * loss 
        #print('affine2 backward:')
        dw2, dx2, db2 = affine_backward(dsig2, a2cache)
        dsig = sigmoid(dx2) * dx2
        #print('affine backward:')
        dw1, dx1, db1 = affine_backward(dsig, a1cache)
        #Update weight
        dw2 = dw2.reshape(w2.shape)
        dw1 = dw1.reshape(w1.shape)
        #print(dw1)
        mb2 +=  dw2 
        mb1 +=  dw1 
        ub1 +=  db1
        ub2 +=  db2
    lr = args.lr
    if i % 5000 == 0:
        lr /= 2
    w2 -= lr * mb2/4
    w1 -= lr * mb1/4
    b2 -= lr * ub2/4
    b1 -= lr * ub1/4
    print('avg. loss:{}'.format(avg/4))
    if avg >100:
        break
out,_ = affine_forward(x,w1,b1)
out,_  = affine_forward(out,w2,b2)
print(out)
