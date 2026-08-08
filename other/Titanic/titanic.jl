using MLDatasets:Titanic

function get_processed_data(args)
    labels = Titanic.targets()
    features = Titanic.features()

    # Split into training and test sets, 2/3 for training, 1/3 for test.
    train_indices = [1:3:891; 2:3:891]

    X_train = features[:, train_indices]
    y_train = labels[:, train_indices]

    X_test = normed_features[:, 3:3:891]
    y_test = onehot_labels[:, 3:3:891]

    #repeat the data `args.repeat` times
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test, y_test)

    return train_data, test_data
end
