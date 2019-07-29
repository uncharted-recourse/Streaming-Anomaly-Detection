from robust_rcf import robust_rcf
import numpy as np
import pandas as pd
from evaluate import evaluate, anomaly_classification_percentile
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def test_rrcf_simon(data, sample = 0.1):
    # load / prepare data
    df = pd.DataFrame(data, columns = ['Timestamp','Year', 'Month', 'Day of Month', 'Day of Week', 'Hour', 'Minute', 'Seconds', 'Simon Features', 'file'])
    # shuffle values for SIMON training / testing
    df_shuffle = df.sample(frac = 1)
    simon_features = np.array(df_shuffle['Simon Features'].values.tolist())
    labels = (df_shuffle['file'] != 'enron.jsonl').astype(int)

    # break into train / test by oldest / newest
    train_split = int(0.6 * df.shape[0])
    val_split = int(0.3 * df.shape[0] * sample)
    simon_train, y_train = simon_features[:train_split], labels[:train_split]
    simon_val, y_val = simon_features[train_split:train_split + val_split], labels[train_split:train_split + val_split]
    simon_test, y_test = simon_features[train_split + val_split:], labels[train_split + val_split:]

    # print anomalous percentage in train / val / test
    print('There are {} ({} %) anomalous examples in the train set'.format(sum(y_train), 100 * sum(y_train) / len(y_train)))
    print('There are {} ({} %) anomalous examples in the sampled validation set'.format(sum(y_val), 100 * sum(y_val) / len(y_val)))
    print('There are {} ({} %) anomalous examples in the test set'.format(sum(y_test), 100 * sum(y_test) / len(y_test)))

    # test batch anomaly detection on SIMON features
    # initially set num_samples_per_tree based on ratio of anomalies
    tree_size = int((df.shape[0] - sum(labels)) / sum(labels) * 2)
    num_trees = 200

    start_time = time.time()
    print('Fitting batch anomaly detection on training set...')
    clf = robust_rcf(num_trees, tree_size)
    clf.fit_batch(simon_train)
    print('Fitting batch anomaly detection took {} seconds'.format(time.time() - start_time))
    '''
    print('Scoring training set')
    start_time = time.time()
    anom_score = clf.batch_anomaly_scores()
    print('Scoring batch anomaly detection took {} seconds'.format(time.time() - start_time))

    # set threshold as % of anomalies in sample
    # TODO = add function that can do percentile or z-score
    anom_thresh = (len(labels) - sum(labels)) / len(labels) * 100
    anom_pred = anomaly_classification_percentile(anom_score, anom_thresh)
    print("Training Set Evaluation")
    print(evaluate(y_train, anom_pred))
    '''
    # eval on validation set
    print('Scoring validation set')
    start_time = time.time()
    val_anom_score = clf.anomaly_score(simon_val)
    print('Scoring batch anomaly detection took {} seconds on ({} %) of the validation set'.format(time.time() - start_time, 100 * sample))
    val_anom_pred = anomaly_classification_percentile(val_anom_score, anom_thresh)
    print("Validation Set Evaluation")
    print(evaluate(y_val, val_anom_pred))
    
    # test streaming anomaly detection on SIMON features (just validation set)
    print('Fitting / scoring streaming anomaly detection on validation set...')
    start_time = time.time()
    stream_anom_scores = clf.stream_anomaly_scores(simon_val, window_size = 1, new_forest=True)
    print('Fitting / Scoring streaming anomaly detection took {} seconds on ({}%) of the validation set'.format(time.time() - start_time, 100 * sample))
    val_anom_pred = anomaly_classification_percentile(stream_anom_scores, anom_thresh)
    print("Validation Set Evaluation")
    print(evaluate(y_val, val_anom_pred))

def test_rrcf_enron_times(data, sample = 0.1, anom_thresh = 95):
    '''
        Test batch and streaming anomaly detection on just Enron email time features
    '''
    # sort Enron emails by timestamp
    df = pd.DataFrame(data, columns = ['Timestamp','Year', 'Month', 'Day of Month', 'Day of Week', 'Hour', 'Minute', 'Seconds', 'Simon Features', 'file'])
    df = df.loc[df['file'] == 'enron.jsonl'].sort_values(by = 'Timestamp')

    # convert timestamp column to timestamp difference
    df['Timestamp Difference'] = df['Timestamp'].diff()

    # drop non-time columns 
    df.drop(['Timestamp', 'Simon Features', 'file'], axis=1, inplace=True)
    #df = df[['Timestamp Difference']]
    # cast to np array of float values and remove initial timestamp (nan time difference)
    df = df.values.astype(float)[1:]

    # test on sample of training / validation data
    train_split = int(0.6 * df.shape[0] * sample)
    val_split = int(0.3 * df.shape[0] * sample)
    enron_train = df[:train_split]
    enron_val = df[train_split:train_split + val_split]
    plt.hist(enron_train)
    plt.show()
    plt.hist(enron_val)
    plt.show()
    
    # test batch anomaly detection
    tree_size = 100
    num_trees = 100
    start_time = time.time()
    print('Fitting batch anomaly detection on training set...')
    clf = robust_rcf(num_trees, tree_size)
    clf.fit_batch(enron_train)
    print('Fitting batch anomaly detection took {} seconds'.format(time.time() - start_time))
    print('Scoring training set')
    start_time = time.time()
    anom_score = clf.anomaly_score(enron_train)
    print('Scoring batch anomaly detection took {} seconds'.format(time.time() - start_time))

    # set "true" anomalies just based on frequency
    anom_pred = anomaly_classification_percentile(anom_score, anom_thresh)
    anom_true = (enron_train[:,-1] < np.percentile(enron_train[:,-1], 100 - anom_thresh)).astype(int)
    print("Training Set Evaluation")
    print(evaluate(anom_true, anom_pred))

    # eval on validation set
    print('Scoring validation set')
    start_time = time.time()
    val_anom_score = clf.anomaly_score(enron_val)
    print('Scoring batch anomaly detection took {} seconds on ({} %) of the validation set'.format(time.time() - start_time, 100 * sample))
    val_anom_pred = anomaly_classification_percentile(val_anom_score, anom_thresh)
    anom_true = (enron_val[:,-1] < np.percentile(enron_val[:,-1],100 - anom_thresh)).astype(int)
    print("Validation Set Evaluation")
    print(evaluate(anom_true, val_anom_pred))

    # graph results
    colors = ('blue', 'red')
    targets = ('non-anomalous', 'anomalous')
    enron_scaled = MinMaxScaler().fit_transform(enron_train[:,-1].reshape(-1,1)).reshape(-1,)
    pred_indices = (np.where(val_anom_pred == 0), np.where(val_anom_pred == 1))
    pred_data = (enron_scaled[np.where(val_anom_pred == 0)[0]], enron_scaled[np.where(val_anom_pred == 1)[0]])
    plt.subplot(2,1,1)
    for index, dat, color, target in zip(pred_indices, pred_data, colors, targets):
        plt.scatter(index[0], dat, c = color, label = target, s=10)
    plt.legend()
    plt.title('Batch Anomaly Detection on Enron Time Series Data')
    plt.show()

    # test streaming anomaly detection on Enron time features (just validation set)
    print('Fitting / scoring streaming anomaly detection on validation set...')
    start_time = time.time()
    stream_anom_scores = clf.stream_anomaly_scores(enron_val, window_size = 1, new_forest=True)
    print('Fitting / Scoring streaming anomaly detection took {} seconds on ({} %) of the validation set'.format(time.time() - start_time, 100 * sample))
    val_anom_pred = anomaly_classification_percentile(stream_anom_scores, anom_thresh)
    print("Validation Set Evaluation")
    print(evaluate(anom_true, val_anom_pred))
    
    # graph results
    colors = ('blue', 'red')
    targets = ('non-anomalous', 'anomalous')
    enron_scaled = MinMaxScaler().fit_transform(enron_train[:,-1].reshape(-1,1)).reshape(-1,)
    pred_indices = (np.where(val_anom_pred == 0), np.where(val_anom_pred == 1))
    pred_data = (enron_scaled[np.where(val_anom_pred == 0)[0]], enron_scaled[np.where(val_anom_pred == 1)[0]])
    plt.subplot(2,1,1)
    for index, dat, color, target in zip(pred_indices, pred_data, colors, targets):
        plt.scatter(index[0], dat, c = color, label = target, s=10)
    plt.legend()
    plt.title('Batch Anomaly Detection on Enron Time Series Data')
    plt.show()

    # test streaming anomaly detection on Enron time features (just validation set)
    print('Fitting / scoring streaming anomaly detection on validation set...')
    start_time = time.time()
    stream_anom_scores = clf.stream_anomaly_scores(enron_val, window_size = 1, new_forest=True)
    print('Fitting / Scoring streaming anomaly detection took {} seconds on ({} %) of the validation set'.format(time.time() - start_time, 100 * sample))
    val_anom_pred = anomaly_classification_percentile(stream_anom_scores, anom_thresh)
    print("Validation Set Evaluation")
    print(evaluate(anom_true, val_anom_pred))

    # graph results
    colors = ('blue', 'red')
    targets = ('non-anomalous', 'anomalous')
    pred_indices = (np.where(val_anom_pred == 0), np.where(val_anom_pred == 1))
    pred_data = (enron_scaled[np.where(val_anom_pred == 0)[0]], enron_scaled[np.where(val_anom_pred == 1)[0]])
    plt.subplot(2,1,2)
    for index, dat, color, target in zip(pred_indices, pred_data, colors, targets):
        plt.scatter(index[0], dat, c = color, label = target, s=10)
    plt.legend()
    plt.title('Streaming Anomaly Detection on Enron Time Series Data')
    plt.show()

def test_rrcf_enron_jpl_times(data, sample = 0.1):
    '''
        Test batch and streaming anomaly detection on JPL Abuse emails superimposed on Enron
        email distribution over time
    '''

    # graph JPL / Nigerian timestamps vs Enron timestamps
    df = pd.DataFrame(data, columns = ['Timestamp','Year', 'Month', 'Day of Month', 'Day of Week', 'Hour', 'Minute', 'Seconds', 'Simon Features', 'file'])
    plt.hist(df.loc[df['file'] == 'enron.jsonl']['Timestamp'], label = 'non-anomalous')
    plt.hist(df.loc[df['file'] != 'enron.jsonl'][df['file'] != 'nigerian.jsonl']['Timestamp'], label = 'anomalous (JPL)')
    plt.legend()
    plt.title('Comparison of Enron and JPL Timestamps Data')
    plt.show()
# resample timestamps?? superimpose

    # test on sample of training / validation data

    # evaluate

def optimize_tree_size(X_val, y_val, sample = 0.1, num_trees = 100,
                       min = 1, max = 2048, step_size = 1, patience = None):
    '''
        Find tree_size that produces highest validation set accuracy for fixed num_trees
        using batch anomaly detection

        Parameters:
            X_val           validation data features
            y_val           validation data labels
            sample          sample of data to take for evaluation
            num_trees       fixed num_trees HP
            min             min size for tree_size HP
            max             optional max size for tree_size HP
            step_size       step_size for incrementing tree_size HP
            patience        stop evaluation if val accuracy hasn't improved for this many steps
    '''
    acc = 0
    best_acc = acc
    best_index = 0

    anom_thresh = (len(y_val) - sum(y_val)) / len(y_val) * 100
    sample = sample * X_val.shape[0]
    patience_count = 0
    for i in range(min, max + step_size, step_size):
        if patience_count >= patience:
            break
        clf = robust_rcf(num_trees, i)
        clf.fit_batch(X_val[:sample])
        anom_score = clf.batch_anomaly_scores()
        anom_pred = anomaly_classification_percentile(anom_score, anom_thresh)
        acc = accuracy_score(y_val[:sample], anom_pred)
        if acc > best_acc:
            best_acc = acc
            best_index = i
        else:
            patience_count += 1
    print('The best accuracy was {} with tree size {}'.format(best_acc, best_index))
    return best_index

def optimize_num_trees(X_val, y_val, sample = 0.1, tree_size = 256,
                       min = 50, max=1000, step_size = 1, patience = None):
    '''
        Find num_trees that produces highest validation set accuracy for fixed tree_size
        using batch anomaly detection

        Parameters:
            X_val           validation data features
            y_val           validation data labels
            sample          sample of data to take for evaluation
            tree_size       fixed tree_size HP
            min             min size for num_trees HP
            max             optional max size for num_trees HP
            step_size       step_size for incrementing num_trees HP
            patience        stop evaluation if val accuracy hasn't improved for this many steps
    '''
    acc = 0
    best_acc = acc
    best_index = 0

    anom_thresh = (len(y_val) - sum(y_val)) / len(y_val) * 100
    sample = sample * X_val.shape[0]
    patience_count = 0
    for i in range(min, max + step_size, step_size):
        if patience_count >= patience:
            break
        clf = robust_rcf(i, tree_size)
        clf.fit_batch(X_val[:sample])
        anom_score = clf.batch_anomaly_scores()
        anom_pred = anomaly_classification_percentile(anom_score, anom_thresh)
        acc = accuracy_score(y_val[:sample], anom_pred)
        if acc > best_acc:
            best_acc = acc
            best_index = i
        else:
            patience_count += 1
    print('The best accuracy was {} with number of trees {}'.format(best_acc, best_index))
    return best_index

# main method for testing methods
if __name__ == '__main__':
    datapath = 'all_emails_parsed.npz'
    data = np.load(datapath)['all_emails']
    #test_rrcf_simon(data, sample = .1)
    test_rrcf_enron_times(data, sample = .05)
