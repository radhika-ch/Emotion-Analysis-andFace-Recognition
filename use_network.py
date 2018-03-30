
import numpy as np
import tensorflow as tf
import pickle
import os
import tkinter as tk
from tkinter import messagebox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


with open("features.pickle","rb") as f:
	while True:
			try:
				n  = pickle.load(f, encoding='latin1')
			except EOFError:
				break
n = np.array(n)

test_size = 200
test_x = n[-test_size:,0]
test_y = n[-test_size:,1]
a = []
b = []

for xs in test_x:
        a.append((xs-np.mean(xs))/np.std(xs))
for xs in test_y:
		b.append(xs)

test_x = np.array(a)
test_y = np.array(b)

# print(test_x.dtype, test_x.shape)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500



n_classes = 5
batch = 10
# height * width

x_p = tf.placeholder('float')
y_p = tf.placeholder('float')


def neural_network_model_d(data):

		hidden_1_layer = {'f_fum': n_nodes_hl1, 'weights':tf.Variable(tf.truncated_normal(shape = [len(test_x[0]), n_nodes_hl1], stddev = 5e-2)), 'biases' : tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1], stddev = 0.1))}
		hidden_2_layer = {'f_fum': n_nodes_hl2,'weights':tf.Variable(tf.truncated_normal(shape = [n_nodes_hl1, n_nodes_hl2], stddev = 5e-2)), 'biases' : tf.Variable(tf.truncated_normal(shape =[n_nodes_hl2], stddev = 0.1))}
		hidden_3_layer = {'f_fum': n_nodes_hl3,'weights':tf.Variable(tf.truncated_normal(shape = [n_nodes_hl2, n_nodes_hl3], stddev = 5e-2)), 'biases' : tf.Variable(tf.truncated_normal(shape = [n_nodes_hl3], stddev = 0.1))}
		output_layer   = {'f_fum': None ,'weights':tf.Variable(tf.truncated_normal(shape = [n_nodes_hl3, n_classes], stddev = 5e-2)), 'biases' : tf.Variable(tf.truncated_normal(shape = [n_classes], stddev = 0.1))}

		l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
		l1 = tf.nn.relu(l1)

		l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
		l2 = tf.nn.relu(l2)

		# l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
		# l3 = tf.nn.relu(l3)

		output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])
		
		return output

prediction = neural_network_model_d(x_p)
with tf.Session() as session:
	global saver	
	# session.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

def use_network(feature, label):
	global prediction
	labels = ['Neutral','Happy','Sad','Angry', 'Surprised']
	#prediction = neural_network_model_d(x_p)

		
	# print(feature)
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(session,'model_detect1.ckpt')
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y_p,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		#print("Model Accuracy: ", accuracy.eval({x_p:test_x, y_p:test_y}))
		result = (session.run(tf.argmax(prediction.eval(feed_dict={x_p:[feature]}),1)))
		#print("\n\n\n\n")	re
		prob = (prediction.eval(feed_dict={x_p:[feature]}))		
		#print(prob)
		#prob = prob/10000
		#prob = ( np.exp(prob)/np.sum(np.exp(prob)))[0][result[0]]
		#print(session.run(tf.nn.softmax(prob))[0][result[0]]*100,end=" %\n")
		
		#print(labels[result[0]])
		
		#print(prob*100,end=" %\n")
		#root = tk.Tk()
		#root.withdraw()

		print("Results", labels[result[0]],"Label: "+labels[np.argmax(label)])




if __name__ == '__main__' :
	with open('file.pickle',"rb") as f:
		n = pickle.load(f, encoding='latin1' )
for i in range(len(n)):
	n1 = np.array(n[i][0])
	# print(n1)

	n1 = (n1-np.mean(n1))/np.std(n1)
	label = np.array(n[i][1])
	#print(label)
	use_network(n1, label)
