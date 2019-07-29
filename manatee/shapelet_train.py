from Sloth.classify import Shapelets
from Sloth.preprocess import events_to_rates
import pandas as pd
import numpy as np
import os.path
#from evaluate import evaluate
#from preprocess import parse_weekly_timestamps
from manatee.evaluate import evaluate
from manatee.preprocess import parse_weekly_timestamps
import matplotlib.pyplot as plt
import pickle
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from keras.models import load_model
from keras.optimizers import Adam, Adagrad, RMSprop

def data_augmentation(X_train, y_train, random_seed = 0):
    '''
        augment samples of less frequent classes
        so that all classes have same number of samples
        random sampling with replacement
    '''
    values, counts = np.unique(y_train, return_counts = True)
    max_samples = counts.max()
    initial = X_train.shape[0]

    for val in values:
        ixs = np.where(y_train == val)[0]
        np.random.seed(random_seed)
        rand_ixs = np.random.choice(y_train[ixs].shape[0], max_samples - len(ixs), replace = True)
        X_train = np.append(X_train, X_train[ixs][rand_ixs], axis = 0)
        y_train = np.append(y_train, y_train[ixs][rand_ixs], axis = 0)
        print("Augmented class {} with {} randomly sampled repetitions (with replacement)".format(val, max_samples - len(ixs)))
    print("\nBefore augmentation the training dataset had {} samples".format(initial))
    print("After augmentation the training dataset has {} samples".format(X_train.shape[0]))
    return X_train, y_train

def data_augmentation_with_noise():
    '''
        augment samples of less frequent classes (with noise)
        so that all classes have same number of samples
    '''
    pass

def shapelet_sizes_grid_search(series_size = 240*60, num_bins = 300, n_folds = 5, 
    min_points = 5, filter_bandwidth = 2, density = True, epochs=100, length=[.025, .05], num_shapelet_lengths=[6,9,12], 
    num_shapelets = .2, learning_rate=.01, weight_regularizer = .01, random_state = 0):
    '''
        grid search over different shapelet dictionary options
        from Grabocka paper
    '''
    acc = 0
    f1_macro = 0
    f1_weighted = 0
    best_min_l_acc = None
    best_num_l_acc = None
    best_min_l_f1_macro = None
    best_num_l_f1_macro = None
    best_min_l_f1_weighted = None
    best_num_l_f1_weighted = None

    # create rate values if they don't already exist
    dir_path = "kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)
    series_values =  np.load("rate_values/" + dir_path + "/series_values.npy")
    labels =  np.load("rate_values/" + dir_path + "/labels.npy")

    # randomly shuffle before splitting into training / test / val
    np.random.seed(random_state)
    randomize = np.arange(len(series_values))
    np.random.shuffle(randomize)
    series_values = series_values[randomize]
    labels = labels[randomize]
    train_split = int(0.9 * series_values.shape[0])

    # write HP combination results to file
    file = open('hp_shp_sizes_grid_search_results.txt', 'a+')
    file.write('%s,%s,%s,%s,%s\n' % ('Min Length', 'Num Shapelet Lengths', 'Accuracy', 'F1 Macro', 'F1 Weighted'))
    file.close()

    for min_length in length:
        for num_lengths in num_shapelet_lengths:

            # CV
            skf = StratifiedKFold(n_splits= n_folds, shuffle = True)
            val_acc = []
            val_f1_macro = []
            val_fl_weighted = []

            print("Evaluating {} shapelet lengths starting at a minimum length of {}".format(num_lengths, min_length))
            for i, (train, val) in enumerate(skf.split(series_values[:train_split], labels[:train_split])):
                print("Running fold {} of {}".format(i+1, n_folds))
                acc_val, f1_macro_val, f1_weighted_val = train_shapelets(series_values[train].reshape(-1, series_values.shape[1], 1), labels[train],
                    series_size = series_size, num_bins = num_bins, density=density, length = min_length, num_shapelet_lengths=num_lengths,
                    val_data=(series_values[val].reshape(-1, series_values.shape[1], 1), labels[val]))
                val_acc.append(acc_val)
                val_f1_macro.append(f1_macro_val)
                val_fl_weighted.append(f1_weighted_val)

            # write mean values
            file = open('hp_shp_sizes_grid_search_results.txt', 'a+')
            file.write('%s,%s,%s,%s,%s\n' % (min_length, num_lengths, np.mean(val_acc),
                np.mean(val_f1_macro), np.mean(val_fl_weighted)))
            file.close()
            
            if np.mean(val_acc) > acc:
                best_min_l_acc = min_length
                best_num_l_acc = num_lengths
                acc = np.mean(val_acc)
            if np.mean(val_f1_macro) > f1_macro:
                best_min_l_f1_macro = min_length
                best_num_l_f1_macro = num_lengths
                f1_macro = np.mean(val_f1_macro)
            if np.mean(val_fl_weighted) > f1_weighted:
                best_min_l_f1_weighted = min_length
                best_num_l_f1_weighted = num_lengths
                f1_weighted = np.mean(val_fl_weighted)

    # return best result
    print("The best accuracy was {} at min length {} and number of lengths {}".format(acc, best_min_l_acc, best_num_l_acc))
    print("The best f1 macro was {} at min length {} and number of lengths {}".format(f1_macro, best_min_l_f1_macro, best_num_l_acc))
    print("The best f1 weighted was {} at min length {} and number of lengths {}".format(f1_weighted, best_min_l_f1_weighted, best_num_l_f1_weighted))

def batch_events_to_rates(data, index, labels_dict = None, series_size = 60*60, min_points = 10, num_bins = 60, filter_bandwidth = 1, density=True):
    '''
        convert list of event times into rate functions using a gaussian filter.

        parameters:
            data         pandas Series containing event times to convert
            index        index containing labels of different clusters to turn into 
                         unique rate functions
            labels_dict  dictionary of labels (1 per index) to apply to labels
            series_size  length of windows that time series capture, expressed in seconds (default = 24 hours)
            min_points   minimum number of points needed to calculate rate function over series of length
                         series_size
            num_bins     number of bins to subdivide series into
            filter_bandwidth       length of gaussian filter 
            density      whether to generate density rate functions

        return:
            series_values   np array containing series values for each series
            series_times    np array containing time values for each point in each series
            labels          np array containing labels for each series
            val_series_count dictionary containing the number of series parsed for each unique value      
    '''

    # assert that data and index have the same length
    try:
        assert(len(data) == len(index))
    except:
        raise ValueError("The series of event times and the series of indices must be the same length")

    # convert each cluster of time series to rates
    series_values = []
    series_times = []
    labels = []
    series_count = 0
    val_series_count = {}
    avg_event_count = {}
    for val in index.unique():
        val_series_count[val] = 0
        avg_event_count[val] = 0
        event_times = data.loc[index == val]

        # iterate through event times by series size -> only convert to rate function if >= min_points events
        event_index = 0
        while (event_index <= event_times.max()):
            events = event_times[(event_index <= event_times) & (event_times < (event_index + series_size))]
            if len(events) >= min_points:
                rate_vals, rate_times = events_to_rates(events.values.astype(int), num_bins = num_bins, filter_bandwidth = filter_bandwidth,
                        min_time = event_index, max_time = event_index + series_size, density = density)
                series_values.append(rate_vals)
                series_times.append(rate_times)
                if labels_dict is not None:
                    labels.append(labels_dict[val])
                series_count += 1
                val_series_count[val] += 1
                print("Added time series with {} events from cluster {}".format(len(events), val))
            else:
                print("Time series from cluster {} is too short".format(val))
            event_index += series_size
            avg_event_count[val] += len(events)
        print("{} time series were added from cluster: {}".format(val_series_count[val], val))
        if val_series_count[val]:
            print("Time series were added from cluster: {} have an average of {} ecents".format(val, avg_event_count[val] / val_series_count[val]))
    print("\nDataset Summary: \n{} total time series, length = {} hr, sampled {} times".format(series_count, series_size / 60 / 60, num_bins))
    for val in index.unique():
        ct = val_series_count[val]
        print("{} time series ({} %) were added from cluster: {}".format(ct, round(ct / series_count * 100, 1), val))
        if val_series_count[val]:
            print("Time series were added from cluster: {} have an average of {} ecents".format(val, avg_event_count[val] / val_series_count[val]))

    labels = np.array(labels)
    series_values = np.vstack(series_values)
    series_times = np.vstack(series_times)

    # save series values and series times if they don't already exist
    dir_path = "rate_values/kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)
    if not os.path.isfile(dir_path + "/series_values.npy"):
        os.mkdir(dir_path)
        np.save(dir_path + "/series_values.npy", series_values)
        np.save(dir_path + "/series_times.npy", series_times)
        np.save(dir_path + "/labels.npy", labels)
        output = open(dir_path + "/val_series_count.pkl", 'wb')
        pickle.dump(val_series_count, output)
        output.close()
    if len(labels_dict.keys()) > 2:
        if not os.path.isfile(dir_path + "/labels_multi.npy"):
            np.save(dir_path + "/labels_multi.npy", labels)
    else:
        if not os.path.isfile(dir_path + "/labels_binary.npy"):
            np.save(dir_path + "/labels_binary.npy", labels)
    return series_values, series_times, labels, val_series_count

def train_shapelets(X_train, y_train, visualize = False, epochs = 10000, length = 0.05, num_shapelet_lengths = 12,
    num_shapelets = .25, learning_rate = .01, weight_regularizer = .001, batch_size = 256, optimizer = Adam, series_size = 240 * 60, 
    num_bins = 300, density = True, p_threshold = 0.5, transfer = False, val_data = None):

    # shapelet classifier
    source_dir = 'shapelets_bad'
    clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer, 
        batch_size = batch_size, optimizer = optimizer)

    # visualize training data
    if visualize:
        for label in np.unique(y_train):
            for index in np.where(y_train == label)[0][:2]:
                plt.plot(np.arange(X_train.shape[1]), X_train[index])
                time_unit = series_size / num_bins / 60
                if time_unit == 1:
                    plt.xlabel('Minute of the Hour')
                elif time_unit == 0.5:
                    plt.xlabel('Half Minute of the Hour')
                if density:
                    plt.ylabel('Email Density')
                else:
                    plt.ylabel('Emails per Minute')
                if label == 1:
                    plt.title('Example of Anomalous Rate Function')
                else:
                    plt.title('Example of Non-Anomalous Rate Function')
                plt.show()
    
    # split into training and validation sets
    if val_data is None:
        #np.random.seed(0)
        inds = np.arange(X_train.shape[0])
        np.random.shuffle(inds)
        X_train = X_train[inds]
        y_train = y_train[inds]
        val_split = 1 / 3
        val_split = int(val_split * X_train.shape[0])
        X_train, y_train = X_train[:-val_split], y_train[:-val_split]
        X_val, y_val = (X_train[-val_split:], y_train[-val_split:])
    else:
        X_val, y_val = val_data

    # data augmentation
    X_train, y_train = data_augmentation(X_train, y_train)

    # eval shapelet on best fit

    # shapelet classifier
    if not transfer:
        print("\nFitting Shapelet Classifier on {} Training Time Series".format(X_train.shape[0]))
        clf.load_model(num_bins, y_train, "checkpoints/shapelets_bad2019-03-22_19-39-26_45-0.2369.h5")
        #clf.fit(X_train, y_train, source_dir = source_dir, val_data = (X_val, y_val))
    else:
        print("\nFitting Shapelet Classifer on {} Training Time Series. Transfer Learned from Binary Setting".format(X_train.shape[0]))
        model = clf.fit_transfer_model(X_train, y_train, "checkpoints/shapelets2019-03-08_01-19-31_61-0.4997.h5", source_dir = source_dir, val_data = (X_val, y_val))
        
    # evaluate after full training
    y_pred = clf.predict_proba(X_val)
    y_preds, conf = clf.decode(y_pred, p_threshold)
    print('\nEvaluation on Randomly Shuffled Validation Set with {} Validation Time Series'.format(X_val.shape[0]))
    #targets = clf.get_classes()
    evaluate(y_val, y_preds)#, target_names=targets)
    #return accuracy_score(y_val, y_preds), f1_score(y_val, y_preds, average='macro'), f1_score(y_val, y_preds, average='weighted')

    # visualize 
    if visualize:
        print('Visualize Shapelet Classifications')
        rates = X_train[-val_split:]
        y_true = y_train[-val_split:]
        for i in np.arange(3):
            if y_true[i] == 1 and y_preds[i] == 1:
                print('Correct Classification: Anomalous')
            elif y_true[i] == 1 and y_preds[i] == 0:
                print('Incorrect Classification: True = Anomalous, Predicted = Non-Anomalous')
            elif y_true[i] == 0 and y_preds[i] == 1:
                print('Incorrect Classification: True = Non-Anomalous, Predicted = Anomalous')
            elif y_true[i] == 0 and y_preds[i] == 0:
                print('Correct Classification: Non-Anomalous')
            clf.VisualizeShapeletLocations(rates, i)

        """ # Shapelet test - track over time
        track = np.array([])
        for i in range(10):
            track = np.append(track, i * 0.01)
        track = np.append(track, track[::-1])

        # Beginning
        track_beg = np.pad(track, (0, num_bins - len(track)), 'constant')
        track_beg = track_beg.reshape(1, track_beg.shape[0], 1)
        track_pred = clf.predict(track_beg)
        if track_pred:
            print('Classification: Anomalous')
        else:
            print('Classification: Non-Anomalous')
        clf.VisualizeShapeletLocations(track_beg, 0, series_size, num_bins, density)

        # Middle
        l = int((num_bins - len(track)) / 2)
        track_mid = np.pad(track, (l, l), 'constant')
        track_mid = track_mid.reshape(1, track_mid.shape[0], 1)
        track_pred = clf.predict(track_mid)
        if track_pred:
            print('Classification: Anomalous')
        else:
            print('Classification: Non-Anomalous')
        clf.VisualizeShapeletLocations(track_mid, 0, series_size, num_bins, density)

        # End
        track_end = np.pad(track, (num_bins - len(track), 0), 'constant')
        track_end = track_end.reshape(1, track_end.shape[0], 1)
        track_pred = clf.predict(track_end)
        if track_pred:
            print('Classification: Anomalous')
        else:
            print('Classification: Non-Anomalous')
        clf.VisualizeShapeletLocations(track_end, 0, series_size, num_bins, density) """

    # hyperparameter optimization

    # shapelet sizes grid search

    # epoch optimization with best HPs and shapelet sizes

def shapelets_hp_opt(length = .05, num_shapelet_lengths = 12, series_size = 240 * 60, n_folds = 3,
    num_bins = 300, min_points = 5, filter_bandwidth =2, density = True, num_shp = .25, lr = [.001, .01, .1],
    wr = .001, b = 256, opt = Adam , epochs = 1000, random_state = 0):
    ''' 
        grid search over different hyperparameter options (learning rate, weight regularizer, batch size, num_shapelets,
        optimizer) from Grabocka paper

        num_shapelets = [.05,.1,.15,.2,.25] 
        learning_rate = [.001, .01, .1],
        weight_regularizer = [.001,.01, .1], 
        batch_size = [64, 128, 256, 512], 
        optimizer = ['Adam', 'Adagrad', 'RMSprop']
    '''
    acc = 0
    f1_macro = 0
    f1_weighted = 0
    best_val_acc = None
    best_val_f1_macro = None
    best_val_f1_weighted = None

    # load rate values 
    dir_path = "kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)
    series_values =  np.load("rate_values/" + dir_path + "/series_values.npy")
    labels =  np.load("rate_values/" + dir_path + "/labels.npy")
    
    # randomly shuffle before splitting into training / test / val
    np.random.seed(random_state)
    randomize = np.arange(len(series_values))
    np.random.shuffle(randomize)
    series_values = series_values[randomize]
    labels = labels[randomize]
    train_split = int(0.9 * series_values.shape[0])

    # write HP combination results to file
    '''
    file = open('hp_grid_search_results.txt', 'a+')
    file.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % ('Num Shapelets', 'Learning Rate', 'Weight Regularizer', 
        'Optimizer', 'Batch Size', 'Accuracy', 'F1 Macro', 'F1 Weighted'))
    file.close()
    '''
    for value in lr:

        # CV
        skf = StratifiedKFold(n_splits= n_folds, shuffle = True)
        val_acc = []
        val_f1_macro = []
        val_fl_weighted = []

        print("Evaluating num_shp: {}, lr: {}, wr: {}, opt: {}, bs: {}".format(num_shp, value, wr, opt, b))
        for i, (train, val) in enumerate(skf.split(series_values[:train_split], labels[:train_split])):
            print("Running fold {} of {}".format(i+1, n_folds))
            acc_val, f1_macro_val, f1_weighted_val = train_shapelets(series_values[train].reshape(-1, series_values.shape[1], 1), labels[train],
                series_size = series_size, num_bins = num_bins, density=density, length = length, num_shapelet_lengths=num_shapelet_lengths,
                val_data=(series_values[val].reshape(-1, series_values.shape[1], 1), labels[val]), learning_rate = value, 
                weight_regularizer = wr, num_shapelets = num_shp, optimizer = opt, batch_size = b, epochs=epochs)
            val_acc.append(acc_val)
            val_f1_macro.append(f1_macro_val)
            val_fl_weighted.append(f1_weighted_val)

        # write mean values
        file = open('hp_grid_search_results.txt', 'a+')
        file.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (num_shp, value, wr, opt, b, np.mean(val_acc),
            np.mean(val_f1_macro), np.mean(val_fl_weighted)))
        file.close()
        if np.mean(val_acc) > acc:
            best_val_acc = value
            acc = np.mean(val_acc)
        if np.mean(val_f1_macro) > f1_macro:
            best_val_f1_macro = value
            f1_macro = np.mean(val_f1_macro)
        if np.mean(val_fl_weighted) > f1_weighted:
            best_val_f1_weighted = value
            f1_weighted = np.mean(val_fl_weighted)

    # return best result
    print("The best accuracy was {} at value {}".format(acc, best_val_acc))
    print("The best f1 macro was {} at value {}".format(f1_macro, best_val_f1_macro))
    print("The best f1 weighted was {} at value {}".format(f1_weighted, best_val_f1_weighted))

def series_size_cv_grid_search(event_times, index, n_folds =5, min = 15 * 60, max = 120*60, step = 15*60, num_bins = 60, 
    min_points = 10, filter_bandwidth = 1, density = True, epochs=100, length=0.1, num_shapelet_lengths=2, 
    num_shapelets = .2, learning_rate=.01, weight_regularizer = .01):
    '''
        grid search over different series size values with 5 fold cross validation. graph results

        30 minute series size provides best accuracy, f1, support across classes
    '''
    # shapelet classifier
    #clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer)

    acc = []
    f1_macro = []
    f1_weighted = []
    for x in range(min, max, step):

        # create rate values if they don't already exist
        if os.path.isfile("rate_values/kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}/series_values.npy".format(x / 60 / 60, num_bins, min_points, filter_bandwidth, density)):
            dir_path = "kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(x / 60 / 60, num_bins, min_points, filter_bandwidth, density)
            series_values =  np.load("rate_values/" + dir_path + "/series_values.npy")
            labels =  np.load("rate_values/" + dir_path + "/labels.npy")
        else:
            labels_dict = {}
            for val in index.unique():
                if val < 50:
                    labels_dict[val] = 0
                else:
                    labels_dict[val] = 1
            series_values, _, labels, _ = \
                batch_events_to_rates(event_times, index, labels_dict, series_size = x, min_points = min_points, 
                    num_bins = num_bins, filter_bandwidth = filter_bandwidth, density = density)

        skf = StratifiedKFold(n_splits= n_folds, shuffle = True)
        val_acc = []
        val_f1_macro = []
        val_fl_weighted = []

        # randomly shuffle before splitting into training / test / val
        np.random.seed(0)
        randomize = np.arange(len(series_values))
        np.random.shuffle(randomize)
        series_values = series_values[randomize]
        labels = labels[randomize]

        # train
        train_split = int(0.9 * series_values.shape[0])

        print("Evaluating series size {}".format(x))
        for i, (train, val) in enumerate(skf.split(series_values[:train_split], labels[:train_split])):
            print("Running fold {} of {}".format(i+1, n_folds))
            acc_val, f1_macro_val, f1_weighted_val = train_shapelets(series_values[train].reshape(-1, series_values.shape[1], 1), labels[train],
                series_size = x, num_bins = num_bins, density=density,
                val_data=(series_values[val].reshape(-1, series_values.shape[1], 1), labels[val]))
            val_acc.append(acc_val)
            val_f1_macro.append(f1_macro_val)
            val_fl_weighted.append(f1_weighted_val)
        acc.append(np.mean(val_acc))
        f1_macro.append(np.mean(val_f1_macro))
        f1_weighted.append(np.mean(val_fl_weighted))
    
    # graph results
    names = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    for vals, name in zip([acc, f1_macro, f1_weighted], names):
        plt.clf()
        plt.plot(range(min, max, step), vals)
        plt.title(name)
        plt.xlabel('Series Size')
        plt.ylabel(name)
        plt.show()

    # return best result
    x_vals = np.arange(min, max, step)
    print("The best accuracy was {} at series size {}".format(np.amax(acc), x_vals[np.argmax(acc)]))
    print("The best f1 macro was {} at series size {}".format(np.amax(f1_macro), x_vals[np.argmax(f1_macro)]))
    print("The best f1 weighted was {} at series size {}".format(np.amax(f1_weighted), x_vals[np.argmax(f1_weighted)]))
    return np.amax(acc), np.amax(f1_macro), np.amax(f1_weighted)

def num_bins_cv_grid_search(event_times, index, n_folds =5, min = 15, max = 61, step = 15, series_size = 30 * 60, 
    min_points = 10, filter_bandwidth = 1, density = True, epochs=100, length=0.1, num_shapelet_lengths=2, 
    num_shapelets = .2, learning_rate=.01, weight_regularizer = .01):
    '''
        grid search over different series size values with 5 fold cross validation. graph results

        135 bins provides best accuracy, f1, support across classes
    '''
    # shapelet classifier
    #clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer)

    acc = []
    f1_macro = []
    f1_weighted = []
    for x in range(min, max, step):

        # create rate values if they don't already exist
        if os.path.isfile("rate_values/kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}/series_values.npy".format(series_size / 60 / 60, x, min_points, filter_bandwidth, density)):
            dir_path = "kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, x, min_points, filter_bandwidth, density)
            series_values =  np.load("rate_values/" + dir_path + "/series_values.npy")
            labels =  np.load("rate_values/" + dir_path + "/labels.npy")
        else:
            labels_dict = {}
            for val in index.unique():
                if val < 50:
                    labels_dict[val] = 0
                else:
                    labels_dict[val] = 1
            series_values, _, labels, _ = \
                batch_events_to_rates(event_times, index, labels_dict, series_size = series_size, min_points = min_points, 
                    num_bins = x, filter_bandwidth = filter_bandwidth, density = density)

        skf = StratifiedKFold(n_splits= n_folds, shuffle = True)
        val_acc = []
        val_f1_macro = []
        val_fl_weighted = []

        # randomly shuffle before splitting into training / test / val
        np.random.seed(0)
        randomize = np.arange(len(series_values))
        np.random.shuffle(randomize)
        series_values = series_values[randomize]
        labels = labels[randomize]

        # train
        train_split = int(0.9 * series_values.shape[0])

        print("Evaluating number of bins {}".format(x))
        for i, (train, val) in enumerate(skf.split(series_values[:train_split], labels[:train_split])):
            print("Running fold {} of {}".format(i+1, n_folds))
            acc_val, f1_macro_val, f1_weighted_val = train_shapelets(series_values[train].reshape(-1, series_values.shape[1], 1), labels[train],
                series_size = series_size, num_bins = x, density=density,
                val_data=(series_values[val].reshape(-1, series_values.shape[1], 1), labels[val]))
            val_acc.append(acc_val)
            val_f1_macro.append(f1_macro_val)
            val_fl_weighted.append(f1_weighted_val)
        acc.append(np.mean(val_acc))
        f1_macro.append(np.mean(val_f1_macro))
        f1_weighted.append(np.mean(val_fl_weighted))
    
    # graph results
    names = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    for vals, name in zip([acc, f1_macro, f1_weighted], names):
        plt.clf()
        plt.plot(range(min, max, step), vals)
        plt.title(name)
        plt.xlabel('Number of Bins')
        plt.ylabel(name)
        plt.show()

    # return best result
    x_vals = np.arange(min, max, step)
    print("The best accuracy was {} at number of bins {}".format(np.amax(acc), x_vals[np.argmax(acc)]))
    print("The best f1 macro was {} at number of bins {}".format(np.amax(f1_macro), x_vals[np.argmax(f1_macro)]))
    print("The best f1 weighted was {} at number of bins {}".format(np.amax(f1_weighted), x_vals[np.argmax(f1_weighted)]))
    return np.amax(acc), np.amax(f1_macro), np.amax(f1_weighted)

def filter_width_cv_grid_search(event_times, index, n_folds =5, min = 1, max = 5, step = 1, series_size = 30 * 60, 
    min_points = 10, num_bins = 135, density = True, epochs=100, length=0.1, num_shapelet_lengths=2, 
    num_shapelets = .2, learning_rate=.01, weight_regularizer = .01):
    '''
        grid search over different series size values with 5 fold cross validation. graph results

        width = 1 provides best accuracy, f1, support across classes
    '''
    # shapelet classifier
    #clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer)

    acc = []
    f1_macro = []
    f1_weighted = []
    for x in range(min, max, step):

        # create rate values if they don't already exist
        if os.path.isfile("rate_values/kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}/series_values.npy".format(series_size / 60 / 60, num_bins, min_points, x, density)):
            dir_path = "kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, min_points, x, density)
            series_values =  np.load("rate_values/" + dir_path + "/series_values.npy")
            labels =  np.load("rate_values/" + dir_path + "/labels.npy")
        else:
            labels_dict = {}
            for val in index.unique():
                if val < 50:
                    labels_dict[val] = 0
                else:
                    labels_dict[val] = 1
            series_values, _, labels, _ = \
                batch_events_to_rates(event_times, index, labels_dict, series_size = series_size, min_points = min_points, 
                    num_bins = num_bins, filter_bandwidth = x, density = density)

        skf = StratifiedKFold(n_splits= n_folds, shuffle = True)
        val_acc = []
        val_f1_macro = []
        val_fl_weighted = []

        # randomly shuffle before splitting into training / test / val
        np.random.seed(0)
        randomize = np.arange(len(series_values))
        np.random.shuffle(randomize)
        series_values = series_values[randomize]
        labels = labels[randomize]

        # train
        train_split = int(0.9 * series_values.shape[0])

        print("Evaluating filter bandwidth {}".format(x))
        for i, (train, val) in enumerate(skf.split(series_values[:train_split], labels[:train_split])):
            print("Running fold {} of {}".format(i+1, n_folds))
            acc_val, f1_macro_val, f1_weighted_val = train_shapelets(series_values[train].reshape(-1, series_values.shape[1], 1), labels[train],
                series_size = series_size, num_bins = num_bins, density=density,
                val_data=(series_values[val].reshape(-1, series_values.shape[1], 1), labels[val]))
            val_acc.append(acc_val)
            val_f1_macro.append(f1_macro_val)
            val_fl_weighted.append(f1_weighted_val)
        acc.append(np.mean(val_acc))
        f1_macro.append(np.mean(val_f1_macro))
        f1_weighted.append(np.mean(val_fl_weighted))
    
    # graph results
    names = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    for vals, name in zip([acc, f1_macro, f1_weighted], names):
        plt.clf()
        plt.plot(range(min, max, step), vals)
        plt.title(name)
        plt.xlabel('Filter Bandwidth')
        plt.ylabel(name)
        plt.show()

    # return best result
    x_vals = np.arange(min, max, step)
    print("The best accuracy was {} at filter bandwidth {}".format(np.amax(acc), x_vals[np.argmax(acc)]))
    print("The best f1 macro was {} at filter bandwidth {}".format(np.amax(f1_macro), x_vals[np.argmax(f1_macro)]))
    print("The best f1 weighted was {} at filter bandwidth {}".format(np.amax(f1_weighted), x_vals[np.argmax(f1_weighted)]))
    return np.amax(acc), np.amax(f1_macro), np.amax(f1_weighted)

def min_points_cv_grid_search(event_times, index, n_folds =5, min = 5, max = 26, step = 5, series_size = 30 * 60, 
    filter_bandwidth = 1, num_bins = 135, density = True, epochs=100, length=0.1, num_shapelet_lengths=2, 
    num_shapelets = .2, learning_rate=.01, weight_regularizer = .01):
    '''
        grid search over different series size values with 5 fold cross validation. graph results

        5 bins (10 bins might be more faithful representation) provides best accuracy, f1, support across classes
    '''
    # shapelet classifier
    #clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer)

    acc = []
    f1_macro = []
    f1_weighted = []
    for x in range(min, max, step):

        # create rate values if they don't already exist
        if os.path.isfile("rate_values/kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}/series_values.npy".format(series_size / 60 / 60, num_bins, x, filter_bandwidth, density)):
            dir_path = "kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, x, filter_bandwidth, density)
            series_values =  np.load("rate_values/" + dir_path + "/series_values.npy")
            labels =  np.load("rate_values/" + dir_path + "/labels.npy")
        else:
            labels_dict = {}
            for val in index.unique():
                if val < 50:
                    labels_dict[val] = 0
                else:
                    labels_dict[val] = 1
            series_values, _, labels, _ = \
                batch_events_to_rates(event_times, index, labels_dict, series_size = series_size, min_points = x, 
                    num_bins = num_bins, filter_bandwidth = filter_bandwidth, density = density)

        skf = StratifiedKFold(n_splits= n_folds, shuffle = True)
        val_acc = []
        val_f1_macro = []
        val_fl_weighted = []

        # randomly shuffle before splitting into training / test / val
        np.random.seed(0)
        randomize = np.arange(len(series_values))
        np.random.shuffle(randomize)
        series_values = series_values[randomize]
        labels = labels[randomize]

        # train
        train_split = int(0.9 * series_values.shape[0])

        print("Evaluating minimum number of points {}".format(x))
        for i, (train, val) in enumerate(skf.split(series_values[:train_split], labels[:train_split])):
            print("Running fold {} of {}".format(i+1, n_folds))
            acc_val, f1_macro_val, f1_weighted_val = train_shapelets(series_values[train].reshape(-1, series_values.shape[1], 1), labels[train],
                series_size = series_size, num_bins = num_bins, density=density,
                val_data=(series_values[val].reshape(-1, series_values.shape[1], 1), labels[val]))
            val_acc.append(acc_val)
            val_f1_macro.append(f1_macro_val)
            val_fl_weighted.append(f1_weighted_val)
        acc.append(np.mean(val_acc))
        f1_macro.append(np.mean(val_f1_macro))
        f1_weighted.append(np.mean(val_fl_weighted))
    
    # graph results
    names = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    for vals, name in zip([acc, f1_macro, f1_weighted], names):
        plt.clf()
        plt.plot(range(min, max, step), vals)
        plt.title(name)
        plt.xlabel('Min Points')
        plt.ylabel(name)
        plt.show()

    # return best result
    x_vals = np.arange(min, max, step)
    print("The best accuracy was {} at minimum number of points {}".format(np.amax(acc), x_vals[np.argmax(acc)]))
    print("The best f1 macro was {} at minimum number of points {}".format(np.amax(f1_macro), x_vals[np.argmax(f1_macro)]))
    print("The best f1 weighted was {} at minimum number of points {}".format(np.amax(f1_weighted), x_vals[np.argmax(f1_weighted)]))
    return np.amax(acc), np.amax(f1_macro), np.amax(f1_weighted)

# main method for training methods
if __name__ == '__main__':
    series_size = 240 * 60
    num_bins = 300
    min_points = 5
    filter_bandwidth = 2
    density = True
    data = pd.read_pickle('../../all_emails_kmeans_clustered.pkl')
    data = parse_weekly_timestamps(data)    

    # 5 fold CV on series size
    '''
    num_bins_cv_grid_search(data['Weekly Timestamp'], data['kmeans'], min = 300, max = 421, step = 30, 
        series_size=series_size, min_points=min_points, filter_bandwidth=filter_bandwidth, density=density)
    
    shapelet_sizes_grid_search(series_size=series_size, filter_bandwidth=filter_bandwidth, 
       num_bins=num_bins, density=density, min_points=min_points)
    '''
    shapelets_hp_opt()
    '''
    # EDA events / series
    
    labels_dict = {}
    for val in data['file'].unique():
        if val == 'enron.jsonl':
            labels_dict[val] = 0
        else:
            labels_dict[val] = 1
    batch_events_to_rates(data['Weekly Timestamp'], data['file'], labels_dict, series_size = series_size, min_points = min_points, 
        num_bins = num_bins, filter_bandwidth = filter_bandwidth, density = density)
    '''