import numpy as np 

synaptic_weights = 2*np.random.random((3, 1))-1

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
	return x * (1 - x)	

def backpropagation(inputs, error, partial, synaptic_weights):
	ajust = np.dot(inputs.T, error*deriv_sigmoid(partial))#ajust synaptic_weights
	synaptic_weights += ajust	

def calculate_error(outputs, partial):
	return (outputs - partial)#calculate error, how much data is loss

def train(synaptic_weights, inputs, outputs, train_rate): 
	for x in range(1,train_rate):
		sum = np.dot(inputs, synaptic_weights)#sum synaptic_weights and inputs
		partial = sigmoid(sum)#activaction function
		error = calculate_error(outputs, partial)
		if(x%1000 == 0): print "error rate %s: " %(np.average(error))
		backpropagation(inputs, error, partial, synaptic_weights)

def test(trained_weigths, inputs):
	return sigmoid(np.dot(inputs, trained_weigths))

train_inputs = np.array([
	[0, 0, 1],
	[0, 1, 0],
	[1, 0, 0]
])

outputs = np.array([
	[0, 
	 1,
	 1
	 ]
]).T

train(synaptic_weights=synaptic_weights, inputs=train_inputs, outputs=outputs, train_rate=10000)

print "predict [1, 0, 1]: should return value close to 0"
print test(synaptic_weights, [1, 0, 1])
