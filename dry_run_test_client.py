#
# Test GRPC client code for NK Streaming Anomaly Detection Classifier
#
#

from __future__ import print_function
import logging
import json
import time
import os
import re
import nltk.data
import time
import json
import pandas as pd
import numpy as np
import configparser
from datetime import datetime, timedelta
import grpc
import logging
import grapevine_pb2
import grapevine_pb2_grpc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
    return int(time.mktime(timestamp.timetuple()))

def parse_emails(N=10000000,datapath=None, filename = 'file'):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(datapath) as data_file:
        data_JSONL_lines = data_file.readlines()
    timestamps = []
    messages = []
    for line in data_JSONL_lines:
        content = json.loads(line)
        t = parse_timestamp(content['date'])
        if filename == 'recourse-attacks.jsonl':
            label = 'true' 
        elif filename == 'ian.jsonl' or filename == 'paul.jsonl' or filename == 'wayne.jsonl' \
            or filename == 'chris.jsonl' or filename == 'christine.jsonl':
            label = 'false'
        else:
            label = 'historical'

        # save email in protobuf message form
        message = grapevine_pb2.Message(
            raw=label,
            language='Field not used. Raw field used to transmit ground truth for evaluation',
            created_at=t, 
            text=content['body'],
        )
        if filename == 'recourse-attacks.jsonl':
            addresses = ['wayne.m.burke@jpl.nasa.gov', 
                            'chris.a.mattmann@jpl.nasa.gov', 
                            'christine.hillsgrove@jpl.nasa.gov',
                            'paul.m.ramirez@jpl.nasa.gov',
                            'ian.colwell@jpl.nasa.gov']
        else:
           addresses = [a['address'] for a in content['to'] + content['cc'] + content['bcc']]
        message.to_account.extend(addresses)
        message.urls.extend(content["urls"])

        # save timestamp and email
        messages.append(message)
        timestamps.append(t)
    return messages, timestamps

def traverse_test_data(datapath):    
    logging.debug(f'Parsing historical emails as text from raw json files...\n')
    emails = []
    all_timestamps = []
    for path, _, files in os.walk(datapath):
        for file in files:
            if re.match(".*.jsonl$", file):
                fullpath = os.path.join(path, file)
                messages, timestamps = parse_emails(datapath = fullpath, filename = file)
                emails.extend(messages)
                all_timestamps.extend(timestamps)
                print(f'Parsed {len(messages)} unique messages under {fullpath} data')

    # sort emails by timestamps
    sorted_emails = np.array(emails)[np.argsort(all_timestamps)]
    #sorted_emails = [email for _,email in sorted(zip(all_timestamps,emails))]
    return sorted_emails

def run():

    channel = grpc.insecure_channel('localhost:' + GRPC_PORT)
    stub = grapevine_pb2_grpc.ClassifierStub(channel)

    # combine all dry run json files and sort by time
    sorted_emails = traverse_test_data('training_data')

    # pass to server
    ground_truth = []
    preds = []
    for idx, email in enumerate(sorted_emails):
        classification = stub.Classify(email)
        pred = classification.prediction
        if email.raw == 'true' or email.raw == 'false':
            ground_truth.append(email.raw)
            preds.append(pred)
        if email.raw == pred:
            print(f'CORRECT: pred = {pred}, truth = {email.raw}, confidence = {classification.confidence}')
        elif email.raw == 'historical':
            print(f'HISTORICAL: pred = {pred}, truth = benign??, confidence = {classification.confidence}')

        else:
            print(f'INCORRECT: pred = {pred}, truth = {email.raw}, confidence = {classification.confidence}')
    
    #evaluate against saved labels
    tn, fp, fn, tp = confusion_matrix(ground_truth, preds).ravel()
    print(f'True Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'True Negatives: {tn}')
    print(f'False Negatives: {fn}')
    print(classification_report(ground_truth, preds, target_names=['false', 'true']))

if __name__ == '__main__':
    logging.basicConfig() # purpose?
    config = configparser.ConfigParser()
    config.read('config.ini')
    port_config = config['DEFAULT']['port_config']
    print("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    run()