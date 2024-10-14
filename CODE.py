import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(2, 4))
weights_hidden_output = np.random.uniform(size=(4, 1))
learning_rate = 0.5

for epoch in range(10000):
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output)
    final_output = sigmoid(final_input)
    
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)
    hidden_error = d_output.dot(weights_hidden_output.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output) 
    
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

print("Output sau khi huấn luyện:")
print(final_output)