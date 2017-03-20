from numpy import exp, array, random, dot, average

class NeuralNetwork():
	def __init__(self):
		random.seed(1)
  		self.synaptic_weights =  2 * random.random((3, 1)) - 1 #Generate random weigths between 0 and 1 positive or negative
  		print "Random synaptic weights" 
  		print self.synaptic_weights

	def sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1 - x)		

	def train(self, train_inputs, right_output, n_iterations):
		for i in range(n_iterations):
			partial_result = self.calculate(train_inputs)			
			error = right_output - partial_result
			if(i%1000 == 0): print "Error rate: %s %s" %(average(error),"%") #Track error rate

			adjustment = dot(train_inputs.T, error * self.sigmoid_derivative(partial_result)) 
			# Adjust the weights.
			self.synaptic_weights += adjustment


	# The neural network calculates.
	def calculate(self, inputs):
		# Pass inputs through our neural network (our single neuron).		
		sum = dot(inputs, self.synaptic_weights)
		return self.sigmoid(sum)

neuron = NeuralNetwork()

training_set_inputs = array([
	[0, 0, 1],
	[0, 1, 0],
	[1, 0, 0],
	[0, 0, 1] 
])

training_set_outputs = array([
	[0, 
	 1,
	 1,
	 0]
]).T

neuron.train(train_inputs=training_set_inputs, right_output=training_set_outputs, n_iterations=10000)

print "Weights after training: "
print neuron.synaptic_weights

print "Test new situation [1, 0, 0] -> ?: should return value close to 1"
print neuron.calculate(array([1, 0, 0]))

