def create_five_nns(input_size, hidden_size, act = None):
    """
    Creates 5 neural networks to be used as a baseline in determining the influence model depth & width has on performance.
    :param input_size: input layer size
    :param hidden_size: list of hidden layer sizes
    :param act: activation function to use for each layer
    :return: list of model_info hash tables
    """
    act = ['relu'] if not act else [act]                             # default activation = 'relu'
    nns = []                                                         # list of model info hash tables
    model_info = {}                                                  # hash tables storing model information
    model_info['Hidden layers'] = [hidden_size]
    model_info['Input size'] = input_size
    model_info['Activations'] = act
    model_info['Optimization'] = 'adadelta'
    model_info["Learning rate"] = .005
    model_info["Batch size"] = 32
    model_info["Preprocessing"] = 'Standard'
    model_info2, model_info3, model_info4, model_info5 = model_info.copy(), model_info.copy(), model_info.copy(), model_info.copy()

    model_info["Name"] = 'Shallow NN'                                 # build shallow nn
    nns.append(model_info)

    model_info2['Hidden layers'] = [hidden_size] * 3                  # build medium nn
    model_info2['Activations'] = act * 3
    model_info2["Name"] = 'Medium NN'
    nns.append(model_info2)

    model_info3['Hidden layers'] = [hidden_size] * 6                  # build deep nn
    model_info3['Activations'] = act * 6
    model_info3["Name"] = 'Deep NN 1'
    nns.append(model_info3)

    model_info4['Hidden layers'] = [hidden_size] * 11                 # build really deep nn
    model_info4['Activations'] = act * 11
    model_info4["Name"] = 'Deep NN 2'
    nns.append(model_info4)

    model_info5['Hidden layers'] = [hidden_size] * 20                   # build realllllly deep nn
    model_info5['Activations'] = act * 20
    model_info5["Name"] = 'Deep NN 3'
    nns.append(model_info5)
    return nns
