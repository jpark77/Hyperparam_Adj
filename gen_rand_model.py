def generate_random_model():
    optimization_methods = ['adagrad', 'rmsprop', 'adadelta', 'adam', 'adamax', 'nadam']      # possible optimization methods
    activation_functions = ['sigmoid', 'relu', 'tanh']          # possible activation functions
    batch_sizes = [16, 32, 64, 128, 256, 512]                   # possible batch sizes
    range_hidden_units = range(5, 250)                          # range of possible hidden units
    model_info = {}                                             # create hash table
    same_units = np.random.choice([0, 1], p=[1/5, 4/5])         # dictates whether all hidden layers will have the same number of units
    same_act_fun = np.random.choice([0, 1], p=[1/10, 9/10])     # will each hidden layer have the same activation function?
    really_deep = np.random.rand()
    range_layers = range(1, 10) if really_deep < 0.8 else range(6, 20)          # 80% of time constrain number of hidden layers between 1 - 10, 20% of time permit really deep architectures
    num_layers = np.random.choice(range_layers, p=[.1, .2, .2, .2, .05, .05, .05, .1, .05]) if really_deep < 0.8 else np.random.choice(range_layers)    # choose number of layers
    model_info["Activations"] = [np.random.choice(activation_functions, p = [0.25, 0.5, 0.25])] * num_layers if same_act_fun else [np.random.choice(activation_functions, p = [0.25, 0.5, 0.25]) for _ in range(num_layers)] # choose activation functions
    model_info["Hidden layers"] = [np.random.choice(range_hidden_units)] * num_layers if same_units else [np.random.choice(range_hidden_units) for _ in range(num_layers)]  # create hidden layers
    model_info["Optimization"] = np.random.choice(optimization_methods)         # choose an optimization method at random
    model_info["Batch size"] = np.random.choice(batch_sizes)                    # choose batch size
    model_info["Learning rate"] = 10 ** (-4 * np.random.rand())                 # choose a learning rate on a logarithmic scale
    model_info["Training threshold"] = 0.5                                      # set threshold for training
    return model_info