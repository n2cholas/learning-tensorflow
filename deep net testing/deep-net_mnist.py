import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
HL = hidden layer
input --> weight --> HL 1 (activation function) --> weights --> HL 2 (activation function) --> weights --> output layer

compare output to intended output --> cost function (cross entropy)
use optimization function (optimizer) --> minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) 

'''
one_hot: basically means one element will be one (hot) 

e.g. 10 classes:
0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 9]
'''

n_nodes_hl1 = 500 #number of nodes for hidden layer 1
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 #go through 100 images (test cases) at a time

'''
matrix is height x width
you don't have to define shape in the placeholder
defining it makes debugging a lot easier because 
tensorflow will throw an error if something that's
not the right shape is passed into it
'''
x = tf.placeholder('float', [None, 784]) # x is input data, none because no height (just vector)
y = tf.placeholder('float') #label of the data

def neural_network_model(data):
    #creates a tensor(array) of random weights and biases for hidden layer 1
    #Reminder: input_data * weights + biases = labels
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer   = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases' : tf.Variable(tf.random_normal([n_classes]))}
    
    #Remember: input_data*weights + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) #activation function (this one's rectilinear)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2) #activation function (this one's rectilinear)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3) #activation function (this one's rectilinear)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
    prediction = neural_network_model(x) #passes input data through neural net and returns one hot array
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)); #computes cost with prediction and y

    #AdamOptimizer also has a parameter learning_rate, but that's 0.001 by default which is good  so excluded here
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #cycles of feed forward and backprop
    n_epochs = 10 #number of epochs

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
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #argmax returns index of maximum values in these arrays
        #basically this line tells us if prediction and y are equal

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)