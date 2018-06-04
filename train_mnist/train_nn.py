import numpy as np
import matplotlib.pyplot as plt
from lib.mnist import load_mnist
from lib.network import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(path='dataset', one_hot_label='True', flatten='False')

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_size = x_train.shape[0]
training_epochs = 16
batch_size = 100
total_batch = int(train_size/batch_size)

train_loss_list = []

for epoch in range(training_epochs):
    avg_loss = 0
    for i in range(total_batch):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        loss, _ = network.train(x_batch, t_batch)
        avg_loss += loss
    avg_loss = avg_loss / total_batch
    train_loss_list.append(avg_loss)
    print('Epoch:', '%04d' %(epoch + 1), 'Avg Loss=', '{:.9f}'.format(avg_loss))
print("Learning finished")
print("Accuracy: "+str(network.accuracy(x_test, t_test)*100))

# Test
# print(np.argmax(network.predict(x_test[0])))
# print(t_test[0])

# Plot
plt.plot(train_loss_list)
plt.xlabel('Epoch')
plt.ylabel('Avg Loss')
plt.show()


# Save network
#network.save_network()

