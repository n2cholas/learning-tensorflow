import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2) #far more efficient than x1*x2

print(result) #prints the tensor object (defines a model, doesn't do the operation)

#One way to create a session:
sess = tf.Session() #gives us a session for tensorflow
print(sess.run(result)) #actually does the operation and prints result
sess.close()

#Recommended way to make a session:
with tf.Session() as sess: #automatically closes session
    output = sess.run(result) #can also store in variable
    print(output)

print(output) #scope works