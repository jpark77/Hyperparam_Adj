
def build_nn(model_info):
    """
    This function builds and compiles a NN given a hash table of the model's parameters.
    :param model_info:
    :return:
    """

    try:
        if model_info["Regularization"] == "l2":                                # if we're using L2 regularization
            lambda_ = model_info['Reg param']                                   # get lambda parameter
            batch_norm, keep_prob = False, False                                # set other regularization tactics

        elif model_info['Regularization'] == 'Batch norm':                      # batch normalization regularization
            lambda_ = 0
            batch_norm = model_info['Reg param']                                # get param
            keep_prob = False
            if batch_norm not in ['before', 'after']:                           # ensure we have a valid reg param
                raise ValueError

        elif model_info['Regularization'] == 'Dropout':                         # Dropout regularization
            lambda_, batch_norm = 0, False
            keep_prob = model_info['Reg param']
    except:
        lambda_, batch_norm, keep_prob = 0, False, False                        # if no regularization is being used

    hidden, acts = model_info['Hidden layers'], model_info['Activations']
    model = Sequential(name=model_info['Name'])
    model.add(InputLayer((model_info['Input size'],)))                            # create input layer
    first_hidden = True

    for lay, act, i in zip(hidden, acts, range(len(hidden))):                                          # create all the hidden layers
        if lambda_ > 0:                                                         # if we're doing L2 regularization
            if not first_hidden:
                model.add(Dense(lay, activation=act, W_regularizer=l2(lambda_), input_shape=(hidden[i - 1],)))    # add additional layers
            else:
                model.add(Dense(lay, activation=act, W_regularizer=l2(lambda_), input_shape=(model_info['Input size'],)))
                first_hidden = False
        else:                                                                   # if we're not regularizing
            if not first_hidden:
                model.add(Dense(lay, input_shape=(hidden[i-1], )))              # add un-regularized layers
            else:
                model.add(Dense(lay, input_shape=(model_info['Input size'],)))  # if its first layer, connect it to the input layer
                first_hidden = False

        if batch_norm == 'before':
            model.add(BatchNormalization(input_shape=(lay,)))               # add batch normalization layer

        model.add(Activation(act))                                          # activation layer is part of the hidden layer

        if batch_norm == 'after':
            model.add(BatchNormalization(input_shape=(lay,)))               # add batch normalization layer

        if keep_prob:
            model.add(Dropout(keep_prob, input_shape=(lay,)))               # dropout layer

    # --------- Adding Output Layer -------------
    model.add(Dense(1, input_shape=(hidden[-1], )))                             # add output layer
    if batch_norm == 'before':                                                  # if we're using batch norm regularization
        model.add(BatchNormalization(input_shape=(hidden[-1],)))
    model.add(Activation('sigmoid'))                                            # apply output layer activation
    if batch_norm == 'after':
        model.add(BatchNormalization(input_shape=(hidden[-1],)))                # adding batch norm layer

    if model_info['Optimization'] == 'adagrad':                                 # setting an optimization method
        opt = optimizers.Adagrad(lr = model_info["Learning rate"])
    elif model_info['Optimization'] == 'rmsprop':
        opt = optimizers.RMSprop(lr = model_info["Learning rate"])
    elif model_info['Optimization'] == 'adadelta':
        opt = optimizers.Adadelta()
    elif model_info['Optimization'] == 'adamax':
        opt = optimizers.Adamax(lr = model_info["Learning rate"])
    else:
        opt = optimizers.Nadam(lr = model_info["Learning rate"])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # compile model

    return model