import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Numpy XOR MLP Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
args = parser.parse_args()
np.random.seed(0)
def sigmoid(x):
    sig_x = 1.0/(1.0 + np.exp(-x))
    return sig_x, x
def sigmoid_backward(dout, cache):
    x = cache
    f,_ = sigmoid(x)
    dx = (1-f) * f * dout
    return dx
def affine_forward(x,w,b):
    out = np.dot(x, w) + b
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
w1 = 0.05 * np.random.randn(2,3)
b1 = np.random.randn(3)
#b1 = np.array([-10,30])
w2 = 0.05 * np.random.randn(3)
b2 = np.random.randn(1)
#b2 = np.array([-30])
y = np.array([0,1,1,0])
for i in range(args.epochs):
    #Forward
    a1out, a1cache = affine_forward(x,w1,b1)
    siga1out, sigcache = sigmoid(a1out)
    a2out, a2cache = affine_forward(siga1out,w2,b2)
    #siga2out, sig2cache = sigmoid(a2out)
    #loss
    print('output: {}'.format(a2out))
    loss = np.power(y - a2out,2).reshape(4,1).mean(axis=1)
    #print(loss)
    #loss = (siga2out - y)*(siga2out - y)
    print(loss.mean())
    if loss.mean() > 100 or loss.mean() < 0.001:
        break;
    #Backward
    #dsig = sigmoid_backward(loss,sig2cache)
    a2cache = a2cache[0], a2cache[1].reshape(3,1), a2cache[2]
    dw2, dx2, db2 = affine_backward(loss.reshape(4,1), a2cache)
    dsig = sigmoid_backward(dx2,sigcache)
    dw1, dx1, db1 = affine_backward(dsig, a1cache)
    #print(dw,dx,db)
    #Update weight
    dw2 = dw2.reshape(w2.shape)
    dw1 = dw1.reshape(w1.shape)
    w2 -=  args.lr * dw2
    w1 -=  args.lr * dw1
    
out,_ = affine_forward(x,w1,b1)
out,_ = sigmoid(out)
out,_  = affine_forward(out,w2,b2)
#out,_  = sigmoid(out)
np.set_printoptions(suppress=True)
print(out)
