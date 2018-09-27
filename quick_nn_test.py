def quick_nn_test(model_info, data_dict, save_path):
    model = build_nn(model_info)                                    # use model info to build and compile a nn
    stop = EarlyStopping(patience=5, monitor='acc', verbose=1)      # maintain a max accuracy for a sliding window of 5 epochs. If we cannot breach max accuracy after 15 epochs, cut model off and move on.
    tensorboard_path =save_path + model_info['Name']                # create path for tensorboard callback
    tensorboard = TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=True, write_images=True)              # create tensorboard callback
    save_model = ModelCheckpoint(filepath= save_path + model_info['Name'] + '\\' + model_info['Name'] + '_saved_' + '.h5')  # save model after every epoch

    model.fit(data_dict['Training data'], data_dict['Training labels'], epochs=150,               # fit model
              batch_size=model_info['Batch size'], callbacks=[save_model, stop, tensorboard])     # evaluate train accuracy
    train_acc = model.evaluate(data_dict['Training data'], data_dict['Training labels'],
                               batch_size=model_info['Batch size'], verbose = 0)
    test_acc = model.evaluate(data_dict['Test data'], data_dict['Test labels'],                   # evaluate test accuracy
                              batch_size=model_info['Batch size'], verbose = 0)

                                                                                        # Get Train AUC
    y_pred = model.predict(data_dict['Training data']).ravel()                          # predict on training data
    fpr, tpr, thresholds = roc_curve(data_dict['Training labels'], y_pred)              # compute fpr and tpr
    auc_train = auc(fpr, tpr)                                                           # compute AUC metric
                                                                                        # Get Test AUC
    y_pred = model.predict(data_dict['Test data']).ravel()                              # same as above with test data
    fpr, tpr, thresholds = roc_curve(data_dict['Test labels'], y_pred)                  # compute AUC
    auc_test = auc(fpr, tpr)

    return train_acc, test_acc, auc_train, auc_test