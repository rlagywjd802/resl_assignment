# import sys, os
import tensorflow as tf

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.session = tf.InteractiveSession()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None, output_size])
        self.y_ = tf.placeholder(tf.float32, [None, output_size])
        self.W1, self.b1, self.W2, self.b2, self.out, self.loss, self.train_step = self.build_network()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_network(self):
        # Layer1
        W1 = tf.get_variable("W1", shape=[self.input_size, self.hidden_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([self.hidden_size]))
        Y1 = tf.nn.sigmoid(tf.matmul(self.x, W1)+b1)
        # Layer1
        W2 = tf.get_variable("W2", shape=[self.hidden_size, self.output_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([self.output_size]))
        Y2 = tf.matmul(Y1, W2)+b2
        # Cost
        loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=Y2)
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

        return W1, b1, W2, b2, Y2, loss, train_step

    def predict(self, input):
        if input.size == input.shape[0]:
            input = input.reshape(1, -1)
        return self.out.eval(feed_dict={self.x: input})

    def train(self, x_batch, y_batch):
        loss, train_step = self.session.run([self.loss, self.train_step],
                                            feed_dict={self.x: x_batch, self.y: y_batch})
        return loss, train_step

    def accuracy(self, test_img, test_lbl):
        correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy.eval(feed_dict={self.x: test_img, self.y_: test_lbl})

    def write_weight(self, f_name, weight):
        with open(f_name, 'w+') as f:
            for row in weight:
                for x in row:
                    f.write(str(x)+" ")
                f.write("\n")

    def write_bias(self, f_name, bias):
        with open(f_name, 'w+') as f:
            for x in bias:
                f.write(str(x)+" ")

    def save_network(self, path):
        self.saver.save(self.session, path+"network/two_layer")
        self.write_weight(path+"weight/w1.txt", self.W1.eval())
        self.write_bias(path+"weight/b1.txt", self.b1.eval())
        self.write_weight(path+"weight/w2.txt", self.W2.eval())
        self.write_bias(path+"weight/b2.txt", self.b2.eval())

    def restore_network(self, path):
        self.saver.restore(self.session, path+"network/two_layer")