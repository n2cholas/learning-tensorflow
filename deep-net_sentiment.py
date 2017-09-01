import tensorflow as tf
from create_sentiment_featuresets import create_feature_sets_and_labels #to use the function to process data
import numpy as np
import pickle

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('positive_examples.txt', 'negative_examples.txt') #pickle.load(open("sentiment_set.pickle","rb"))

n_nodes = [500, 500, 500] #number of nodes for each hidden layer
n_layers = len(n_nodes)

n_classes = 2
batch_size = 1000 #go through 100 images (test cases) at a time

'''
matrix is height x width
you don't have to define shape in the placeholder
defining it makes debugging a lot easier because 
tensorflow will throw an error if something that's
not the right shape is passed into it
'''
x = tf.placeholder('float', [None, len(train_x[0])]) # x is input data, none because no height (just vector)
y = tf.placeholder('float') #label of the data

def neural_network_model(data):
    #creates a tensor(array) of random weights and biases for hidden layer 1
    #Reminder: input_data * weights + biases = labels
    hidden_layers = [None]*len(n_nodes) #preallocate list for each hidden layer
    hidden_layers[0] = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes[0]])),
                        'biases' : tf.Variable(tf.random_normal([n_nodes[0]]))} #initialize first hidden layer
    for i in range(n_layers-1): #initialize rest
        hidden_layers[i+1] = {'weights': tf.Variable(tf.random_normal([n_nodes[i], n_nodes[i+1]])),
                              'biases' : tf.Variable(tf.random_normal([n_nodes[i+1]]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes[-1], n_classes])),
                    'biases' : tf.Variable(tf.random_normal([n_classes]))}
    
    l = [None]*len(n_nodes) #preallocate list

    #Remember: input_data*weights + biases
    l[0] = tf.add(tf.matmul(data, hidden_layers[0]['weights']), hidden_layers[0]['biases'])
    l[0] = tf.nn.relu(l[0]) #activation function (this one's rectilinear)
    for i in range(n_layers-1):
        l[i+1] = tf.add(tf.matmul(l[i], hidden_layers[i+1]['weights']), hidden_layers[i+1]['biases'])
        l[i+1] = tf.nn.relu(l[i+1]) #activation function (this one's rectilinear)

    output = tf.add(tf.matmul(l[-1], output_layer['weights']), output_layer['biases'])

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

            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y}) #just runs those two variables
                #don't care about optimizer, so just underscore that
                #want the cost, so capture in variable c
                epoch_loss += c

                i += batch_size
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Epoch', epoch+1, 'completed out of', n_epochs, '  Loss:', epoch_loss, '  Accuracy: ', accuracy.eval({x:test_x, y:test_y}))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #argmax returns index of maximum values in these arrays
        #basically this line tells us if prediction and y are equal

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)