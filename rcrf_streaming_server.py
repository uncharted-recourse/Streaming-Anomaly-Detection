#
# GRPC Server for NK FCN-LSTM Classifier
# 
# Uses GRPC service config in protos/grapevine.proto
# 

from flask import Flask, request
import os
import re
import nltk.data
import time
import json
import pandas as pd
import numpy as np
import configparser
from datetime import datetime, timedelta
import pickle
from manatee import robust_rcf
import grpc
import grapevine_pb2
import grapevine_pb2_grpc
from concurrent import futures
from Simon import Simon 
from Simon.Encoder import Encoder
from Simon.DataGenerator import DataGenerator
from Simon.LengthStandardizer import *
import logging
import sys

# GLOBALS
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEBUG = True # boolean to specify whether or not print DEBUG information

restapp = Flask(__name__)

def check_timestamp_range(timestamps):
    '''
        Check whether timestamp range is greater than one week
    '''
    return True if (max(timestamps) - min(timestamps)).total_seconds() > 604800 else False

def parse_time_features(timestamp, weekly_time):
    '''
        Parse higher-dimensional time information from `timestamp`
    '''
    # Year, Month, Day of Month, Day of Week, Hour, Minutes, Seconds
    if weekly_time:
        return [timestamp.weekday(), timestamp.hour, timestamp.minute, timestamp.second]
    else:
        return [timestamp.hour, timestamp.minute, timestamp.second]

def parse_timestamp(date):
    '''
        Parse timestamp from date
    '''
    # parse timestamp 
    timestamp = datetime.strptime(date[0:19], '%Y-%m-%dT%H:%M:%S')
    if date[19]=='+':
        timestamp-=timedelta(hours=int(date[20:22]), minutes = int(date[23:]))
    elif date[19]=='-':
        timestamp+=timedelta(hours=int(date[20:22]), minutes = int(date[23:]))
    return timestamp

def parse_emails(N=10000000,datapath=None, maxlen = 200, max_cells = 100):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(datapath) as data_file:
        data_JSONL_lines = data_file.readlines()
    # now, build up pandas dataframe of appropriate format for NK email classifier
    idx = 0
    account_ids = {}
    timestamps = []
    for line in data_JSONL_lines:
        
        # generate simon text content
        sample_email = ''
        content = json.loads(line)
        for url in content["urls"]:
            sample_email += url + ' '
        sample_email += content["body"]
        sample_email_sentence = tokenizer.tokenize(sample_email)
        sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] #truncate
        if idx == 0:
            all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
        else:
            all_email_df = pd.concat([all_email_df,pd.DataFrame(sample_email_sentence,columns=['Email '+str(idx)])],axis=1)
        
        # grab timestamp content
        timestamps.append(parse_timestamp(content['date']))

        # match all to, cc, bcc account ids
        for address in content['to'] + content['cc'] + content['bcc']:
            if address['address'] in account_ids.keys():
                account_ids[address['address']].append(idx)
            else:
                account_ids[address['address']] = [idx]

        # increment
        idx = idx+1
        if idx>=N-1:
            break

    return pd.DataFrame.from_records(DataLengthStandardizerRaw(all_email_df,max_cells)), timestamps, account_ids

def traverse_training_data(datapath, model, encoder, maxlen = 200, max_cells = 100, checkpoint_dir = 'deployed_checkpoints/'):    
    logging.debug(f'Parsing historical emails as text from raw json files...\n')
    accounts_to_emails = {}
    for path, _, files in os.walk(datapath):
        for file in files:
            if re.match(".*.jsonl$", file):
                fullpath = os.path.join(path, file)
                df_simon, timestamps, account_ids = parse_emails(datapath = fullpath, maxlen=maxlen, max_cells=max_cells)
                raw_data = np.asarray(df_simon.ix[:max_cells-1,:])
                raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

                # produce simon feature vector
                logging.info(f'producing Simon feature vectors for {fullpath}')
                X = encoder.x_encode(raw_data,maxlen)
                preds = model.predict(X)

                # map simon feature vectors to appropriate inboxes for training
                for account, idxs in account_ids.items():
                    accounts_to_emails[account] = [[timestamps[i] for i in idxs], [preds[i] for i in idxs]]
                logging.info(f'Added {len(account_ids.keys())} unique account channels under {fullpath} data')

    return accounts_to_emails

#-----
class NKStreamingClassifier(grapevine_pb2_grpc.ClassifierServicer):

    def __init__(self):

        logging.info('Beginning initialization of distributed, streaming anomaly detection server')
        begin_time = time.time()

        # rrcf classifier parameters
        # TODO: tune forest parameters
        self.TREE_SIZE = 50
        self.NUM_TREES = 100
        training_data_dir = 'training_data'

        # Simon model parameters
        self.maxlen = 200
        self.max_cells = 100
        checkpoint_dir = 'deployed_checkpoints/'

        # instantiate Simon feature model
        config = Simon({}).load_config(MODEL_OBJECT,checkpoint_dir)
        self.encoder = config['encoder']
        Classifier = Simon(encoder=self.encoder)
        self.model = Classifier.generate_feature_model(self.maxlen, self.max_cells, len(self.encoder.categories), checkpoint_dir, config)

        # check if training data exists
        if len(os.listdir('training_data')) == 0:
            return
        
        # training data folder contains pickled dictionary linking account id to training data
        else:
            # initialize separate rrcf classifier object for each sequence in configuration file
            #self.classifiers = traverse_training_data(training_data_dir, self.model, self.encoder,
            #                maxlen=self.maxlen, max_cells=self.max_cells, checkpoint_dir=checkpoint_dir)

            # pickle parsed emails for testing
            #pickle.dump( self.classifiers, open( "classifiers.pkl", "wb" ) )

            # #load parsed emails
            self.classifiers = pickle.load( open( "classifiers.pkl", "rb" ) )

            for account, train in self.classifiers.items():

                # generate higher d time features for training sets
                # only include weekly time information if span of training set is longer
                weekly_bool = check_timestamp_range(train[0])
                time_feature_list = np.array([parse_time_features(t, weekly_bool) for t in train[0]])
                repeated_time_feature_list = np.repeat(time_feature_list, int(len(train[1][0]) / time_feature_list.shape[1]), axis=1)

                # concatenate with text features
                features = np.concatenate((repeated_time_feature_list, train[1]), axis = 1)

                # train separate rrcf classifier given training data in each sequence 
                start_time = time.time()
                tree_size = self.TREE_SIZE if features.shape[0] >= self.TREE_SIZE else features.shape[0]
                self.classifiers[account] = [robust_rcf(self.NUM_TREES, tree_size), weekly_bool]
                self.classifiers[account][0].fit_batch(features)
                logging.info(f"Time to train account {account} classifier: {time.time()-start_time}")

                # record max anomaly score from training set -> generate threshold for prediction
                ## TODO: tune anomaly threshold
                threshold = ANOMALY_THRESHOLD * self.classifiers[account][0].batch_anomaly_scores().values.max()
                self.classifiers[account].append(threshold)
        
        self.model._make_predict_function()
        logging.info(f'Completed initialization of distributed, streaming anomaly detection server. Total time = {(time.time() - begin_time) / 60} mins')

    # Main classify function
    def Classify(self, request, context):

        # init classifier result object
        result = grapevine_pb2.Classification(
            domain=DOMAIN_OBJECT,
            prediction='false',
            confidence=0.0,
            model="NK_Streaming_Anomaly_Detection",
            version="0.0.1",
            meta=grapevine_pb2.Meta(),
        )

        # get timestamp from input message
        input_time = request.created_at
        if input_time is None:
            return result
        if request.to_account == []:
            return result

        # get text from input message and generate simon feature vector
        sample_email = ''
        for url in request.urls:
            sample_email += url + ' '
        sample_email += request.text
        if (len(sample_email.strip()) == 0) or (sample_email is None):
            return result
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sample_email_sentence = tokenizer.tokenize(sample_email)
        sample_email_sentence = [elem[-self.maxlen:] for elem in sample_email_sentence] #truncate
        all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
        all_email_df = pd.DataFrame.from_records(DataLengthStandardizerRaw(all_email_df,self.max_cells))        
        raw_data = np.asarray(all_email_df.ix[:self.max_cells-1,:])
        raw_data = np.char.lower(np.transpose(raw_data).astype('U'))
        text_features = self.model.predict(self.encoder.x_encode(raw_data,self.maxlen))[0]
        
        # match to_account information with stored classifiers
        preds = []
        weights = []
        for account in request.to_account:            
            if account in self.classifiers.keys():

                # calculate appropriate higher d time features for new point
                time_features = np.array(parse_time_features(datetime.fromtimestamp(input_time), self.classifiers[account][1]))
                repeated_time_feature_list = np.repeat(time_features, int(len(text_features) / time_features.shape[0]), axis=0)

                # concatenate with text features
                features = np.concatenate((repeated_time_feature_list, text_features)).reshape(1,-1)

                # generate prediction / confidence by account
                stream_anomaly_score = self.classifiers[account][0].stream_anomaly_scores(features, 1).iloc[0]
                stationary_anomaly_score = self.classifiers[account][0].anomaly_score(features).iloc[0]
                anomaly_threshold = self.classifiers[account][2]
                if request.raw == 'true':
                    logging.info(f'anomaly threshold: {anomaly_threshold}')
                    logging.info(f'streaming anomaly score: {stream_anomaly_score}')
                    logging.info(f'stationary anomaly score: {stationary_anomaly_score}')

                if anomaly_threshold == 0:
                    preds.append(stationary_anomaly_score - anomaly_threshold)
                elif stationary_anomaly_score > anomaly_threshold:
                    preds.append((stationary_anomaly_score - anomaly_threshold) / anomaly_threshold)
                else:
                    preds.append(1 - (anomaly_threshold - stationary_anomaly_score) / anomaly_threshold)    
                weights.append(self.classifiers[account][0].num_points)        

        # take weighted average of predictions across accounts by confidence
        ## TODO: weight to account more than cc / bcc
        # TODO: weight by tree size instead of num_points??
        if np.sum(weights) == 0.0:
            weighted_pred = 0.0
        else:
            weighted_pred = np.sum([w * p for w,p in zip(weights, preds)]) / np.sum(weights)

        if weighted_pred > 0.5:
            result.prediction='true'
            result.confidence=min((weighted_pred - 0.5) / 0.5, 1.0)
        else:
            result.prediction='false'
            result.confidence=min((0.5 - weighted_pred) / 0.5, 1.0)

        # log info on false negatives 
        if request.raw == 'true':
            logging.info(f'anomaly score: {preds}')
            logging.info(f'List of preds: {preds}')
            logging.info(f'List of weights: {weights}')
            logging.info(f'Weighted pred: {weighted_pred}')
            logging.info(f"Classification result is (class / confidence): {result.prediction} / {result.confidence}")
        return result

#-----
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ClassifierServicer_to_server(NKStreamingClassifier(), server)
    server.add_insecure_port('[::]:' + GRPC_PORT)
    server.start()
    restapp.run()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

@restapp.route("/healthcheck")
def health():
    return "HEALTHY"

if __name__ == '__main__':

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'    
    )
    logging.getLogger('')

    config = configparser.ConfigParser()
    config.read('config.ini')
    modelName = config['DEFAULT']['modelName']
    logging.info("using model " + modelName + " ...")
    global MODEL_OBJECT
    MODEL_OBJECT = modelName
    domain = config['DEFAULT']['domain']
    logging.info("using domain " + domain + " ...")
    global DOMAIN_OBJECT
    DOMAIN_OBJECT = domain
    port_config = config['DEFAULT']['port_config']
    logging.info("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    anomaly_threshold = config['DEFAULT']['anomaly_threshold']
    logging.info("using anomaly_threshold " + anomaly_threshold + " ...")
    global ANOMALY_THRESHOLD
    ANOMALY_THRESHOLD = float(anomaly_threshold)
    serve()