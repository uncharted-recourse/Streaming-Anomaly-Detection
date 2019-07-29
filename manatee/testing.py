
    
import os
import re
import json
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report, f1_score
logging.basicConfig(level=logging.DEBUG)
import pickle
import random
import nltk.data
from Simon import Simon 
from Simon.Encoder import Encoder
from Simon.DataGenerator import DataGenerator
from Simon.LengthStandardizer import *

from robust_rcf import robust_rcf

def parse_timestamp(date):
    '''
        Parse timestamp and higher-dimensional time information from `email` containing corpus of emails
    '''
    # parse timestamp 
    timestamp = datetime.strptime(date[0:19], '%Y-%m-%dT%H:%M:%S')
    if date[19]=='+':
        timestamp-=timedelta(hours=int(date[20:22]), minutes = int(date[23:]))
    elif date[19]=='-':
        timestamp+=timedelta(hours=int(date[20:22]), minutes = int(date[23:]))
    return timestamp

def parse_email(email, filename, index, length, accounts_to_emails):
    '''
        Parse timestamp, and higher-dimensional time information from `email` 
    '''
    logging.debug(f'Parsing email {index} of {length} from file: {filename}')
    if filename not in accounts_to_emails.keys():
        accounts_to_emails[filename] = [parse_timestamp(email['date'])]
    else:
        accounts_to_emails[filename].append(parse_timestamp(email['date']))

def traverse_files(datapath):
    logging.debug(f'Parsing historical emails from raw json files...\n')
    accounts_to_emails = {}
    for path, _, files in os.walk(datapath):
        for file in files:
            if re.match(".*.jsonl$", file):
                fullpath = os.path.join(path, file)
                with open(fullpath) as data_file:
                    emails = data_file.readlines()
                    for index, email in enumerate(emails):
                        parse_email(json.loads(email), file, index, len(emails), accounts_to_emails)
    return accounts_to_emails

def check_timestamp_range(timestamps):
    '''
        Check whether timestamp range is greater than one week
    '''
    return True if (max(timestamps) - min(timestamps)).seconds > 604800 else False

def parse_time_features(timestamp, weekly_time):
    '''
        Parse higher-dimensional time information from `timestamp`
    '''
    # Year, Month, Day of Month, Day of Week, Hour, Minutes, Seconds
    if weekly_time:
        return [timestamp.weekday(), timestamp.hour, timestamp.minute, timestamp.second]
        #return [int(time.mktime(timestamp.timetuple())), timestamp.weekday(), timestamp.hour, timestamp.minute, timestamp.second]
    else:
        return [timestamp.hour, timestamp.minute, timestamp.second]
        #return [int(time.mktime(timestamp.timetuple())), timestamp.hour, timestamp.minute, timestamp.second]

def traverse_files_simon(datapath):    
    logging.debug(f'Parsing historical emails as text from raw json files...\n')
    accounts_to_emails = {}
    accounts_to_times = {}
    maxlen = 200
    max_cells = 100
    for path, _, files in os.walk(datapath):
        for file in files:
            if re.match(".*.jsonl$", file):
                fullpath = os.path.join(path, file)
                df, accounts_to_times = parse_emails_simon(accounts_to_times, datapath = fullpath)
                raw_data = np.asarray(df.ix[:max_cells-1,:])
                raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

                # produce simon feature vector
                print(f'producing Simon feature vectors for {fullpath}')
                checkpoint_dir = "../../NK-email-classifier/deployed_checkpoints/"
                config = Simon({}).load_config('text-class.10-0.42.pkl',checkpoint_dir)
                X = np.ones((raw_data.shape[0], max_cells, maxlen), dtype=np.int64) * -1
                encoder = config['encoder']
                Classifier = Simon(encoder=encoder)
                model = Classifier.generate_feature_model(maxlen, max_cells, len(encoder.categories) ,checkpoint_dir, config)
                accounts_to_emails[file] = model.predict(X)
    return accounts_to_emails, accounts_to_times

def parse_emails_simon(accounts_to_times, N=10000000,datapath=None):
    maxlen = 200
    max_cells = 100
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(datapath) as data_file:
        data_JSONL_lines = data_file.readlines()
    # now, build up pandas dataframe of appropriate format for NK email classifier
    idx = 0
    for line in data_JSONL_lines:
        print(idx)
        sample_email = ''
        content = json.loads(line)
        if datapath not in accounts_to_times.keys():
            accounts_to_times[datapath] = [parse_timestamp(content['date'])]
        else:
            accounts_to_times[datapath].append(parse_timestamp(content['date']))
        for url in content["urls"]:
            sample_email += url + ' '
        sample_email += content["body"]
        sample_email_sentence = tokenizer.tokenize(sample_email)
        sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] #truncate
        if idx == 0:
            all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
        else:
            all_email_df = pd.concat([all_email_df,pd.DataFrame(sample_email_sentence,columns=['Email '+str(idx)])],axis=1)
        idx = idx+1
        if idx>=N-1:
            break
    return pd.DataFrame.from_records(DataLengthStandardizerRaw(all_email_df,max_cells)), accounts_to_times

# # walk through datapath and parse all jsonl files from dry run and historical data for timestamps 
# # create dictionary of account ids to list of timestamps
# # dry_run_email_dict = traverse_files('../../LSTM-FCN/dry_run_data')
# # historical_email_dict = traverse_files('../../LSTM-FCN/historical_dry_run_data')

# # # pickle parsed emails
# # pickle.dump( dry_run_email_dict, open( "dry_run_email_dict.pkl", "wb" ) )
# # pickle.dump( historical_email_dict, open( "historical_email_dict.pkl", "wb" ) )

# #load parsed emails
# dry_run_email_dict = pickle.load( open( "dry_run_email_dict.pkl", "rb" ) )
# historical_email_dict = pickle.load( open( "historical_email_dict.pkl", "rb" ) )

#SIMON email text parsing, pickling, loading
# dry_run_email_dict, dry_run_times_dict= traverse_files_simon('../../LSTM-FCN/dry_run_data')
# historical_email_dict, historical_times_dict = traverse_files_simon('../../LSTM-FCN/historical_dry_run_data')

# # pickle parsed emails
# pickle.dump( dry_run_email_dict, open( "dry_run_email_dict_simon.pkl", "wb" ) )
# pickle.dump( historical_email_dict, open( "historical_email_dict_simon.pkl", "wb" ) )

#load parsed emails
dry_run_email_dict = pickle.load( open( "dry_run_email_dict_simon.pkl", "rb" ) )
historical_email_dict = pickle.load( open( "historical_email_dict_simon.pkl", "rb" ) )
dry_run_times_dict = pickle.load( open( "dry_run_email_dict.pkl", "rb" ) )
historical_times_dict = pickle.load( open( "historical_email_dict.pkl", "rb" ) )

# rrcf classifier parameters
# TODO: set these parameters based on size of training set -> different for every account?
TREE_SIZE = 50
NUM_TREES = 100

## TODO: Try scaling data before fitting tree algorithm??

# train rrcf classifiers on historical data
account_dict = {'historical_christine.jsonl': 'christine.jsonl', 
                'historical_chris.jsonl': 'chris.jsonl', 
                'historical_ian.jsonl': 'ian.jsonl', 
                'historical_paul.jsonl': 'paul.jsonl',
                'historical_wayne.jsonl': 'wayne.jsonl'}
classifier_dict = {}

# average_f1_list = []
# for NUM_TREES in range(50, 350,50):
average_f1 = []
average_threshold = []
for account, train in historical_email_dict.items():
    time_train = historical_times_dict[account]
    best_f1 = 0
    best_threshold = 0
    weekly_bool = check_timestamp_range(time_train)

    # generate higher d time features for training sets - only include weekly time information if span of training set is longer
    time_feature_list = np.array([parse_time_features(t, weekly_bool) for t in time_train], dtype='float')
    #repeated_time_feature_list = np.repeat(time_feature_list, int(train.shape[1] / time_feature_list.shape[1]), axis=1)
    features = np.concatenate((time_feature_list, train), axis = 1)
    features = features[np.argsort(historical_times_dict[account])]
    #features[:,0] = np.concatenate((np.array([0]), np.diff(features[:,0])))
    #features = np.concatenate((np.repeat(features[:,:time_feature_list.shape[1]], 
    #            int(train.shape[1] / time_feature_list.shape[1]), axis=1), features[:,time_feature_list.shape[1]:]), axis=1)
    # train separate rrcf classifier given training data in each sequence 
    print(f'Beginning to train account {account} classifier on {features.shape[0]} feature vectors of size {features.shape[1]}')
    start_time = time.time()
    classifier_dict[account] = robust_rcf(NUM_TREES, TREE_SIZE)
    #classifier_dict[account].fit_batch(time_feature_list)
    classifier_dict[account].fit_batch(features)
    print(f"Time to train account {account} classifier: {time.time()-start_time}")

    # get anomaly scores on training data and visualize distribution
    start_time = time.time()
    train_anom_scores = classifier_dict[account].batch_anomaly_scores().values
    print(f"Time to predict anomaly scores on training set for account {account}: {time.time()-start_time}")
    plt.hist(train_anom_scores, 25, density=True, alpha = 0.75)
    # plt.xlabel('Anomaly Scores')
    # plt.ylabel('Probability')
    # plt.title(f'Distribution of {train.shape[0]} Train Anomaly Scores for account {account}')
    # plt.show()

    # add recourse attack emails to each account in dry_run dict - sort by timestamp to reproduce real use case
    times = dry_run_times_dict[account_dict[account]] + dry_run_times_dict['recourse-attacks.jsonl']
    email_features = np.concatenate((dry_run_email_dict[account_dict[account]], dry_run_email_dict['recourse-attacks.jsonl']), axis=0)
    time_feature_list = np.array([parse_time_features(t, weekly_bool) for t in times], dtype='float')
    #repeated_time_feature_list = np.repeat(time_feature_list, int(train.shape[1] / time_feature_list.shape[1]), axis=1)
    features = np.concatenate((time_feature_list, email_features), axis=1)
    features = features[np.argsort(times)]
    #features[:,0] = np.concatenate((np.array([0]), np.diff(features[:,0])))

    # maintain labeled list to evaluate predictions
    labels = [1 if timestamp in dry_run_times_dict['recourse-attacks.jsonl'] else 0 for timestamp in times]
    # predict anomaly scores on test set in training set, visualize distribution
    start_time = time.time()
    #test_anom_scores = classifier_dict[account].stream_anomaly_scores(time_feature_list, 1).values
    test_anom_scores = classifier_dict[account].stream_anomaly_scores(features, 1).values
    print(f"Time to predict anomaly scores per label ({len(labels)} labels) on testing set for account {account}: {(time.time()-start_time) / len(labels)}")
    # plt.hist(test_anom_scores, 25, density=True, alpha = 0.75)
    # plt.xlabel('Anomaly Scores')
    # plt.ylabel('Probability')
    # plt.title(f'Distribution of {features.shape[0]} Test Anomaly Scores for account {account}')

    # evaluate predicted anomaly scores at different thresholds
    max_anom_score = int(train_anom_scores.max())
    min_anom_score = int(train_anom_scores.min())
    step = max(int((test_anom_scores.max() - test_anom_scores.min()) / 20), 1)
    for threshold in range(min_anom_score, max_anom_score + step - 1, step):
        #print(f'Classification report for threshold: {threshold} on account {account}')
        anom_scores = [1 if score > threshold else 0 for score in test_anom_scores]
        #print(classification_report(labels, anom_scores, target_names=['non-anomalous', 'anomalous']))
        f1 = f1_score(labels, anom_scores, average="micro")
        #print(f'F1 score for threshold {threshold / max_anom_score} = {f1}')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold / max_anom_score
    print(f'Best F1 = {best_f1} at threshold: {best_threshold}')
    average_f1.append(best_f1)
    average_threshold.append(best_threshold)
    #plt.show()
print(f'Mean F1 at num trees = {np.mean(average_f1)}')
print(f'Median Threshold at num trees = {np.median(average_threshold)}')
#     average_f1_list.append(np.mean(average_f1))
# plt.clf()
# plt.plot(np.arange(50, 350,50), average_f1_list)
# plt.title('Tree Size HP')
# plt.show()

