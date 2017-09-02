'''
Recurrent Neural Network
- uses previous output as input for the next activation in the sequeince

LSTM - Long Short term memory
- For the recurrent data (coming in), there is a keep/forget gate
- For input, we ask what we want to add
- For the recurrent data going out, there is a what to send gate
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn  #this is for newer version of tensorflow

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) 

n_classes = 10
n_epochs = 3 #number of epochs, cycles of feed forward and backprop
batch_size = 128 #go through 100 images (test cases) at a time
chunk_size = 28 #chunks of 28 pixels at a time (for sequence)
n_chunks = 28
rnn_size = 128 #instead having all those layers, just rnn size
#this rnn size is also quite small

x = tf.placeholder('float', [None, n_chunks, chunk_size]) # x is input data, none because no height (just vector)
y = tf.placeholder('float') #label of the data

def recurrent_neural_net(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases' : tf.Variable(tf.random_normal([n_classes]))} #initialize first hidden layer

    #note: matrix operations basically act the same in tensorflow as in numpy
    x = tf.transpose(x, [1, 0, 2]) 
    '''
    Example of transpose:

    x = np.ones((1, 2, 3))
    y = np.transpose(x, (1, 0, 2))

    x = [[
        [1 1 1],
        [1 1 1]
    ]]
    y = [
        [[1 1 1]],
        [[1 1 1]]
    ]
    '''
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32) #each cell has these at every recurrence

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_net(x) #passes input data through neural net and returns one hot array
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)); #computes cost with prediction and y

    #AdamOptimizer also has a parameter learning_rate, but that's 0.001 by default which is good  so excluded here
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): #cycles based on batch size and total size
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) #conveniently gives you number of examples you need at a time
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size)) #this is batch size because one batch at a time
                
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y}) #just runs those two variables
                #don't care about optimizer, so just underscore that
                #want the cost, so capture in variable c
                epoch_loss += c
            print('Epoch', epoch + 1, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #argmax returns index of maximum values in these arrays
        #basically this line tells us if prediction and y are equal

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #this one's -1 because this is one image at a time
        print('Accuracy: ', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)