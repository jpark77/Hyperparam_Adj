"""This section of code allows us to create and test many neural networks and save the results of a quick 
test into a CSV file. Once that CSV file has been created, we will continue to add results onto the existing 
file."""

rapid_testing_path = 'YOUR PATH HERE'  
data_path = 'YOUR DATA PATH'

try:                                                                        # try to load existing csv
    rapid_mlp_results = pd.read_csv(rapid_testing_path + 'Results.csv')
    index = rapid_mlp_results.shape[1]
except:                                                                     # if no csv exists yet, create a DF
    rapid_mlp_results = pd.DataFrame(columns=['Model', 'Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC',
                                              'Preprocessing', 'Batch size', 'Learn Rate', 'Optimization', 'Activations',
                                              'Hidden layers', 'Regularization'])
    index = 0

og_one_hot = np.array(pd.read_csv(data_path))                     # load one hot data

model_info = {}                                                     # create model_info dicts for all the models we want to test
model_info['Hidden layers'] = [100] * 6                             # specifies the number of hidden units per layer
model_info['Input size'] = og_one_hot.shape[1] - 1                  # input data size
model_info['Activations'] = ['relu'] * 6                            # activation function for each layer
model_info['Optimization'] = 'adadelta'                             # optimization method
model_info["Learning rate"] = .005                                  # learning rate for optimization method
model_info["Batch size"] = 32
model_info["Preprocessing"] = 'Standard'                            # specifies the preprocessing method to be used

model_0 = model_info.copy()                                         # create model 0
model_0['Name'] = 'Model0'

model_1 = model_info.copy()                                         # create model 1
model_1['Hidden layers'] = [110] * 3
model_1['Name'] = 'Model1'

model_2 = model_info.copy()                                         # try best model so far with several regularization parameter values
model_2['Hidden layers'] = [110] * 6
model_2['Name'] = 'Model2'
model_2['Regularization'] = 'l2'
model_2['Reg param'] = 0.0005

model_3 = model_info.copy()
model_3['Hidden layers'] = [110] * 6
model_3['Name'] = 'Model3'
model_3['Regularization'] = 'l2'
model_3['Reg param'] = 0.05

# .... create more models ....

#-------------- REGULARIZATION OPTIONS -------------
#   L2 Regularization:      Regularization: 'l2',           Reg param: lambda value
#   Dropout:                Regularization: 'Dropout',      Reg param: keep_prob
#   Batch normalization:    Regularization: 'Batch norm',   Reg param: 'before' or 'after'


models = [model_0, model_1, model_2]                                  # make a list of model_info hash tables

column_list = ['Model', 'Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC', 'Preprocessing',
               'Batch size', 'Learn Rate', 'Optimization', 'Activations', 'Hidden layers',
               'Regularization', 'Reg Param']

for model in models:                                                                                          # for each model_info in list of models to test, test model and record results
    train_data, labels = preprocess_data(og_one_hot, model['Preprocessing'], True)                            # preprocess raw data
    data_dict = split_data(0.9, 0, np.concatenate((train_data, labels.reshape(29999, 1)), axis=1))             # split data
    train_acc, test_acc, auc_train, auc_test = quick_nn_test(model, data_dict, save_path=rapid_testing_path)  # quickly assess model

    try:
        reg = model['Regularization']                                             # set regularization parameters if given
        reg_param = model['Reg param']
    except:
        reg = "None"                                                              # else set NULL params
        reg_param = 'NA'

    val_lis = [model['Name'], train_acc[1], test_acc[1], auc_train, auc_test, model['Preprocessing'],
                model["Batch size"], model["Learning rate"], model["Optimization"], str(model["Activations"]),
                str(model["Hidden layers"]), reg, reg_param]

    df_dict = {}
    for col, val in zip(column_list, val_lis):                                    # create df dict to append to csv file
        df_dict[col] = val

    df = pd.DataFrame(df_dict, index=[index])
    rapid_mlp_results = rapid_mlp_results.append(df, ignore_index=False)
    rapid_mlp_results.to_csv(rapid_testing_path + "Results.csv", index=False)