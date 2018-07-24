'''
Feed Forward NN
input > weights > hidden layer 1 (activation function) > weights > hidden layer 2 
(activation  function) > weights > output layer

compare output to target output > cost/ loss function
optimization function(optimizer) > minimize the cost (AdamOptimizer...SGD, AdaGrad)

backpropagation 
feed forward + backprop = epoch  

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  

# Multi class 10 classes, 0-9
'''
one hot means one elements/pixel is on and the other is off.
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
'''
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # data break into 100 batches

# height x weight
x = tf.placeholder('float',[None, 784]) # x is input_data
y = tf.placeholder('float')             # y is a label

def neural_network_model(data):
    
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    # (input_data * weights) + biases
    
    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']) , hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)         # relu is activation function
    
    
    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']) , hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)         # relu is activation function
    
    
    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']) , hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)         # relu is activation function
    
    
    output = tf.add(tf.matmul(l3,output_layer['weights']) , output_layer['biases'])
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # cycles feed forward + backprop
    epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x_epoch, y_epoch = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: x_epoch , y: y_epoch})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)
         
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
train_neural_network(x)    



























 