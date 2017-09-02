'''
Convolutional Neural Networks:

Typically 3 hidden layers (2 convolutional, 1 fully connected): 
input --> (conv --> pool) --> (conv --> pool) --> fully connected layer --> output

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) 


n_classes = 10
batch_size = 128 #go through 100 images (test cases) at a time
n_epochs = 3 #number of epochs, cycles of feed forward and backprop

x = tf.placeholder('float', [None, 784]) # x is input data, none because no height (just vector)
y = tf.placeholder('float') #label of the data

#Set up for dropout, need more data to be accurate
#keep_prob at 0.8 for training, but change to 1.0 for actaul useage of NN
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W): #pass data and weights
    #this doesn't have to be it's own function, but you could make this more complex in time
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #stride makes it moves one pixel at a time

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #                           size of window      movement of window

def convolutional_neural_net(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])), #5x5 convolution, 1 input, 32 features/outputs
               'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])), #fully connected
               #7x7 because each step reduces dimensions by half (original 28x28) (bc stride of 2)
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}
    
    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])), #fully connected
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1]) #reshape to flat 28x28
    
    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'] )
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'] )
    conv2 = maxpool2d(conv2)

    fc  = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    #fc = tf.nn.dropout(fc, keep_rate) #80% of neurons will be kept
    
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_net(x) #passes input data through neural net and returns one hot array
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)); #computes cost with prediction and y

    #AdamOptimizer also has a parameter learning_rate, but that's 0.001 by default which is good  so excluded here
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): #cycles based on batch size and total size
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) #conveniently gives you number of examples you need at a time
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y}) #just runs those two variables
                #don't care about optimizer, so just underscore that
                #want the cost, so capture in variable c
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #argmax returns index of maximum values in these arrays
        #basically this line tells us if prediction and y are equal

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)