#
# GRPC Server for Streaming Anomaly Detection Server
# 
# Uses GRPC service config in protos/grapevine.proto
# 

from flask import Flask, request
import nltk.data
import time
import pandas as pd
import numpy as np
import configparser
from datetime import datetime
from manatee import robust_rcf
import grpc
import grapevine_pb2
import grapevine_pb2_grpc
from concurrent import futures
from Simon import Simon 
from Simon.Encoder import Encoder
from Simon.LengthStandardizer import *
import logging
from scipy import stats

# GLOBALS
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEBUG = True # boolean to specify whether or not print DEBUG information

restapp = Flask(__name__)

def parse_time_features(timestamp, weekly_time):
    '''
        Parse higher-dimensional time information from `timestamp`
    '''
    # Year, Month, Day of Month, Day of Week, Hour, Minutes, Seconds
    if weekly_time:
        return [timestamp.weekday(), timestamp.hour, timestamp.minute, timestamp.second]
    else:
        return [timestamp.hour, timestamp.minute, timestamp.second]

#-----
class NKStreamingClassifier(grapevine_pb2_grpc.ClassifierServicer):

    def __init__(self):

        #logging.info('Beginning initialization of distributed, streaming anomaly detection server')
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
        self.model._make_predict_function()

        # dictionary to store separate models by account
        self.classifiers = {}

        #threshold = ANOMALY_THRESHOLD * self.classifiers[account][0].batch_anomaly_scores().values.max()
        
        #logging.info(f'Completed initialization of distributed, streaming anomaly detection server. Total time = {(time.time() - begin_time) / 60} mins')

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
            if account not in self.classifiers.keys():
                # 1: rcrf model, 2. anomaly scores overtime
                self.classifiers[account] = [robust_rcf(self.NUM_TREES, self.TREE_SIZE), []]

            # calculate appropriate higher d time features for new point
            time_features = np.array(parse_time_features(datetime.fromtimestamp(input_time), True))
            #repeated_time_feature_list = np.repeat(time_features, int(len(text_features) / time_features.shape[0]), axis=0)

            # concatenate with text features
            features = np.concatenate((time_features, text_features)).reshape(1,-1)

            # generate prediction and add to tracked list
            stream_anomaly_score = self.classifiers[account][0].stream_anomaly_scores(features, 1).iloc[0]
            if stream_anomaly_score == 0.0:
                percentile_score = 0.0
            else:
                percentile_score = stats.percentileofscore(self.classifiers[account][1], stream_anomaly_score) / 100
            self.classifiers[account][1].append(stream_anomaly_score)
            preds.append(percentile_score)

            # TODO: weight by num_points or tree size??
            weights.append(self.classifiers[account][0].num_points)   

            # DEBUG
            #if DEBUG: and request.raw == 'true':
                #logging.info(f'streaming anomaly score: {stream_anomaly_score}')
                #logging.info(f'account anomaly scores: {self.classifiers[account][1]}')
                #logging.info(f'percentile anomaly score: {percentile_score}')
 
        # take weighted average of predictions across accounts by confidence
        ## TODO: weight to account more than cc / bcc
        if np.sum(weights) == 0.0:
            weighted_pred = 0.0
        else:
            weighted_pred = np.sum([w * p for w,p in zip(weights, preds)]) / np.sum(weights)
        if weighted_pred >= ANOMALY_THRESHOLD:
            result.prediction = 'true'
            result.confidence = (weighted_pred - ANOMALY_THRESHOLD) / (1 - ANOMALY_THRESHOLD)
        else: 
            result.prediction = 'false'
            result.confidence = (ANOMALY_THRESHOLD - weighted_pred) / ANOMALY_THRESHOLD

        # log info on false negatives 
        #if DEBUG: and request.raw == 'true':
            #logging.info(f'List of preds: {preds}')
            #logging.info(f'List of weights: {weights}')
            #logging.info(f'Weighted pred: {weighted_pred}')
            #logging.info(f"Classification result is (class / confidence): {result.prediction} / {result.confidence}")
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
    # logging.basicConfig(
    #     level=logging.info,
    #     format='%(asctime)s - %(levelname)s - %(message)s'    
    # )
    # logging.getLogger('')

    config = configparser.ConfigParser()
    config.read('config.ini')
    modelName = config['DEFAULT']['modelName']
    #logging.info("using model " + modelName + " ...")
    global MODEL_OBJECT
    MODEL_OBJECT = modelName
    domain = config['DEFAULT']['domain']
    #logging.info("using domain " + domain + " ...")
    global DOMAIN_OBJECT
    DOMAIN_OBJECT = domain
    port_config = config['DEFAULT']['port_config']
    #logging.info("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    anomaly_threshold = config['DEFAULT']['anomaly_threshold']
    #logging.info("using anomaly_threshold " + anomaly_threshold + " ...")
    global ANOMALY_THRESHOLD
    ANOMALY_THRESHOLD = float(anomaly_threshold)
    serve()