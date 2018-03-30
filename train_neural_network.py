import numpy as np
import os
import tensorflow as tf
import pickle
#from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)



os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


with open("features.pickle","rb") as f:
	while True:
			try:
				n = pickle.load(f)
			except EOFError:
				break
n = np.array(n)
print(n.shape)
m = np.array(n[0][0])

test_size = 200
train_x = n[:-test_size,0]
train_y = n[:-test_size,1]
test_x = n[-test_size:,0]
test_y = n[-test_size:,1]
print(test_x.shape)
a = []
b = []
for xs in train_x:
	a.append((xs-np.mean(xs))/np.std(xs))
for xs in train_y:
	b.append(xs)
train_x = np.array(a)
train_y = np.array(b)


a = []
b = []
for xs in test_x:
	a.append((xs-np.mean(xs))/np.std(xs))

for xs in test_y:
	b.append(xs)
test_x = np.array(a)
test_y = np.array(b)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
print(test_x.shape)


n_classes = 5
batch_size = 80
# height * width

x = tf.placeholder('float', shape=(None, len(train_x[0])))
y = tf.placeholder('float')


def neural_network_model(data):

		hidden_1_layer = {'f_fum': n_nodes_hl1, 'weights':tf.Variable(tf.truncated_normal(shape = [len(train_x[0]), n_nodes_hl1], stddev = 5e-2)), 'biases' : tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1], stddev = 0.1))}
		hidden_2_layer = {'f_fum': n_nodes_hl2,'weights':tf.Variable(tf.truncated_normal(shape = [n_nodes_hl1, n_nodes_hl2], stddev = 5e-2)), 'biases' : tf.Variable(tf.truncated_normal(shape =[n_nodes_hl2], stddev = 0.1))}
		hidden_3_layer = {'f_fum': n_nodes_hl3,'weights':tf.Variable(tf.truncated_normal(shape = [n_nodes_hl2, n_nodes_hl3], stddev = 5e-2)), 'biases' : tf.Variable(tf.truncated_normal(shape = [n_nodes_hl3], stddev = 0.1))}
		output_layer   = {'f_fum': None ,'weights':tf.Variable(tf.truncated_normal(shape = [n_nodes_hl3, n_classes], stddev = 5e-2)), 'biases' : tf.Variable(tf.truncated_normal(shape = [n_classes], stddev = 0.1))}

		l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
		l1 = tf.nn.relu(l1)

		l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
		l2 = tf.nn.relu(l2)

		l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
		l3 = tf.nn.relu(l3)

		output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])
		
		return output




def train_neural_network(x):
		prediction = neural_network_model(x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
		optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

		hm_epochs = 100
		
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			saver = tf.train.Saver()
			saver.restore(sess,'model_detect1.ckpt')
			for epoch in range(hm_epochs):
				epoch_loss = 0
				i = 1
				while i < len(train_x):
					start = i
					end = i + batch_size

					epch_x = np.array(train_x[start:end])
					epch_y = np.array(train_y[start:end])
					_, c = sess.run([optimizer, cost], feed_dict = {x:epch_x, y:epch_y})
					epoch_loss += c

					i += batch_size
				
				print('Epoch ', epoch+1, ' completed out of', hm_epochs, ' Loss: ', epoch_loss)
			saver = tf.train.Saver()
			saver.save(sess, 'model_detect1.ckpt')
			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			print("Accuracy: ", accuracy.eval({x:test_x, y:test_y}))
train_neural_network(x)