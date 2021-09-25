#/usr/bin/python

from itertools import product
from ann import ann
from mnist_loader import load_data_wrapper


train_d, valid_d, test_d = load_data_wrapper()

HLS = [50]
ETA = [0.5, 0.25, 0.125]
def train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10):
    for h, n in product(hls, eta):
        print(f'---- Training 784x{h}x10 ANN, eta = {n}')
        net = ann([784, h, 10])
        net.mini_batch_sgd(train_d, num_epochs, mini_batch_size, n, test_data=test_d)
        print('---- Training complete')

def train_2_hidden_layer_anns(hls=ETA, eta=ETA, mini_batch_size=10, num_epochs=10):
    for h1, h2, n in product(hls, hls, eta):
        print(f'---- Training 784x{h1}x{h2}x10 ANN, eta = {n}')
        net = ann([784, h1, h2, 10])
        net.mini_batch_sgd(train_d, num_epochs, mini_batch_size, n, test_data=test_d)
        print('---- Training complete')


### Uncomment to run
if __name__ == '__main__':
    #train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
    train_2_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
    pass
