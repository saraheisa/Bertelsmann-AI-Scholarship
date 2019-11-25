# Implement the following functions

# Activation (sigmoid) function
def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(( weights[0] * features[0] )+ ( weights[1] * features[1] ) + bias)

# Error (log-loss) formula
def error_formula(y, output):
    return ( -y * np.log(output) ) - (1 - y) * np.log(1-output)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    yhat = output_formula(x, weights, bias)
    err = learnrate * (y-yhat)
    ws = err * x
    bs = bias + diff
    return ws, bs