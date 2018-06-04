import numpy as np
from lib.mnist import load_mnist
from lib.network import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(path='dataset', one_hot_label='True', flatten='True')

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network.restore_network()
print(np.argmax(network.predict(x_test[0])))
print(t_test[0])