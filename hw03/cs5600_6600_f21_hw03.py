#/usr/bin/python

from ann import ann
from mnist_loader import load_data_wrapper


train_d, valid_d, test_d = load_data_wrapper()

HLS = [10, 25, 50]
ETA = [0.5, 0.25, 0.125]
def train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10):
    ### your code here
    pass

def train_2_hidden_layer_anns(hls=ETA, eta=ETA, mini_batch_size=10, num_epochs=10):
    ### your code here
    pass

### Uncomment to run
if __name__ == '__main__':
    #train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
    #train_2_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
    pass
