import os
import re
import json
from datetime import datetime, timedelta
import time
import nltk
import numpy as np
import pandas as pd
#from Simon import *
#from Simon.Encoder import *
#from Simon.DataGenerator import *
#from Simon.LengthStandardizer import *
from keras.models import Model
from keras.layers import Input, Lambda, Convolution1D, Dropout, MaxPooling1D, merge, LSTM, TimeDistributed, Dense
from sklearn.cluster import KMeans

def parse_time_features(timestamp):
    '''
        Parse higher-dimensional time information from `timestamp`
    '''
    # Year, Month, Day of Month, Day of Week, Hour, Minutes, Seconds
    return [int(time.mktime(timestamp.timetuple())), timestamp.year, timestamp.month, timestamp.day, timestamp.weekday(), timestamp.hour, timestamp.minute, timestamp.second]

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
    return parse_time_features(timestamp)


def generate_feature_model():
    '''
        Generate SIMON feature model to calculate representation of body of email
    '''
     # set important parameters
    max_len = 200 # length of each sentence
    max_cells = 100 # maximum number of sentences per email

    DEBUG = True # boolean to specify whether or not print DEBUG information

    checkpoint_dir = "/Users/jeffreygleason 1/Desktop/NewKnowledge/Code/ASED/NK-email-classifier/deployed_checkpoints/"
    execution_config="text-class.17-0.07.pkl"

    # load specified execution configuration
    if execution_config is None:
        raise TypeError
    Classifier = Simon(encoder={}) # dummy text classifier

    config = Classifier.load_config(execution_config, checkpoint_dir)
    data_encoder = config['encoder']
    checkpoint = config['checkpoint']

    '''GENERATE MODEL FOR LAST LAYER OF SIMON FEATURES'''
    
    filter_length = [1, 3, 3]
    nb_filter = [40, 200, 1000]
    pool_length = 2
    # document input
    document = Input(shape=(max_cells, max_len), dtype='int64')
    # sentence input
    in_sentence = Input(shape=(max_len,), dtype='int64')
    # char indices to one hot matrix, 1D sequence to 2D
    embedded = Lambda(Classifier.binarize, output_shape=Classifier.binarize_outshape)(in_sentence)
    # embedded: encodes sentence
    for i in range(len(nb_filter)):
        embedded = Convolution1D(nb_filter=nb_filter[i],
                                    filter_length=filter_length[i],
                                    border_mode='valid',
                                    activation='relu',
                                    init='glorot_normal',
                                    subsample_length=1)(embedded)

        embedded = Dropout(0.1)(embedded)
        embedded = MaxPooling1D(pool_length=pool_length)(embedded)

    forward_sent = LSTM(256, return_sequences=False, dropout_W=0.2,
                    dropout_U=0.2, consume_less='gpu')(embedded)
    backward_sent = LSTM(256, return_sequences=False, dropout_W=0.2,
                    dropout_U=0.2, consume_less='gpu', go_backwards=True)(embedded)

    sent_encode = merge([forward_sent, backward_sent],
                        mode='concat', concat_axis=-1)
    sent_encode = Dropout(0.3)(sent_encode)
    # sentence encoder

    encoder = Model(input=in_sentence, output=sent_encode)

    #print(encoder.summary())
    encoded = TimeDistributed(encoder)(document)

    # encoded: sentences to bi-lstm for document encoding
    forwards = LSTM(128, return_sequences=False, dropout_W=0.2,
                    dropout_U=0.2, consume_less='gpu')(encoded)
    backwards = LSTM(128, return_sequences=False, dropout_W=0.2,
                    dropout_U=0.2, consume_less='gpu', go_backwards=True)(encoded)

    merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
    output_pre = Dropout(0.3)(merged)
    output_pre = Dense(128, activation='relu', name='features')(output_pre)
    output = Dropout(0.3)(output_pre)
    output = Dense(2, activation='softmax')(output)
    model = Model(input=document, output=output)
    Classifier.load_weights(checkpoint,config,model,checkpoint_dir)

    '''GENERATE MODEL FOR INTERMEDIATE LAYER'''
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer('features').output)
    return data_encoder, intermediate_model

def parse_simon_features(sample_email, encoder, intermediate_model):
    '''
        Parse SIMON feature representation of body of email
    '''
    # set important parameters
    max_len = 200 # length of each sentence
    max_cells = 100 # maximum number of sentences per email

    #start_time = time.time()
    #print("DEBUG::sample email (whole, then tokenized into sentences):")
    #print(sample_email)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sample_email_sentence = tokenizer.tokenize(sample_email)
    sample_email_sentence = [elem[-max_len:] for elem in sample_email_sentence] # truncate
    #print(sample_email_sentence)
    all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
    #print("DEBUG::the final shape is:")
    #print(all_email_df.shape)
    all_email_df = all_email_df.astype(str)
    raw_data = np.asarray(all_email_df.ix[:max_cells-1,:]) #truncate to max_cells
    raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

    X = encoder.x_encode(raw_data,max_len)

    '''GENERATE FEATURES FOR EMAIL'''
    y = intermediate_model.predict(X)
    # discard empty column edge case
    y[np.all(all_email_df.isnull(),axis=0)]=0

    #elapsed_time = time.time()-start_time
    #print("Total time for classification is : %.2f sec" % elapsed_time)
    
    return y[0]

def parse_email(email, filename, index, length, encoder, intermediate_model):
    '''
        Parse SIMON feature representation, timestamp, and higher-dimensional time information from `email` 
    '''
    print('Parsing email {} of {} from file: {}'.format(index, length, filename))
    return [*parse_timestamp(email['date']), parse_simon_features(email['body'], encoder, intermediate_model), filename]

def parse_all_emails(datapath):
    '''
        Parse all timestamps from all emails contained in `datapath`
    '''

    # blacklist of email corpuses not to use
    blacklist = ['FalsePositive.jsonl', 'Recon.jsonl', 'Unknown.jsonl']

    # walk through datapath and parse all jsonl files
    all_emails = []
    encoder, intermediate_model = generate_feature_model()
    for path, _, files in os.walk(datapath):
        for file in files:
            if re.match(".*.jsonl$", file) and file not in blacklist:
                fullpath = os.path.join(path, file)
                with open(fullpath) as data_file:
                    emails = data_file.readlines()
                    parsed_emails = [parse_email(json.loads(email), file, index, len(emails), encoder, intermediate_model) for index, email in enumerate(emails)]
                    all_emails.extend(parsed_emails)

    # convert parsed emails to dataframe 
    return pd.DataFrame(all_emails, columns = ['Timestamp','Year', 'Month', 'Day of Month', 'Day of Week', 'Hour', 'Minute', 'Seconds', 'Simon Features', 'file'])

def cluster_emails(datapath, min_cluster_size, min_samples = 1):

    # load data from saved file
    data = np.load(datapath)['all_emails']
    df = pd.DataFrame(data, columns = ['Timestamp','Year', 'Month', 'Day of Month', 'Day of Week', 'Hour', 'Minute', 'Seconds', 'Simon Features', 'file'])

    # instantiate hdbscan clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size= min_cluster_size, min_samples=min_samples)

    # cluster enron and nigerian emails
    df['labels'] = np.nan
    clusterer.fit(pd.DataFrame((df.loc[df['file'] == 'enron.jsonl']['Simon Features']).values.tolist()))
    nlabels = clusterer.labels_.max()
    df.loc[df.index[df['file'] == 'enron.jsonl'], 'labels'] = clusterer.labels_
    clusterer.fit(pd.DataFrame((df.loc[df['file'] == 'nigerian.jsonl']['Simon Features']).values.tolist()))
    labels = clusterer.labels_ + nlabels + 1
    nlabels = labels.max()
    df.loc[df.index[df['file'] == 'nigerian.jsonl'], 'labels'] = labels
    # add cluster lablels for JPL abuse set
    # TODO - could cluster these ourselves??
    for cat in df['file'].unique():
        if cat not in ['enron.jsonl', 'nigerian.jsonl']:
            nlabels += 1
            df.loc[df.index[df['file'] == cat], 'labels'] = nlabels
    return df

def cluster_emails_kmeans(datapath, n_clusters_enron = 50):

    # choose number of kmeans clusters to try to approximately equalize number of events / series 

    # load data from saved file
    data = np.load(datapath)['all_emails']
    df = pd.DataFrame(data, columns = ['Timestamp','Year', 'Month', 'Day of Month', 'Day of Week', 'Hour', 'Minute', 'Seconds', 'Simon Features', 'file'])

    # cluster enron and nigerian emails
    df['kmeans'] = np.nan
    enron_data = pd.DataFrame((df.loc[df['file'] == 'enron.jsonl']['Simon Features']).values.tolist())
    kmeans = KMeans(n_clusters = n_clusters_enron).fit(enron_data)
    nlabels = kmeans.labels_.max()
    df.loc[df.index[df['file'] == 'enron.jsonl'], 'kmeans'] = kmeans.labels_

    '''
    nigerian_data = pd.DataFrame((df.loc[df['file'] == 'nigerian.jsonl']['Simon Features']).values.tolist())
    kmeans_enron.fit(nigerian_data)
    labels = kmeans_enron.predict(nigerian_data)
    nlabels = labels.max()
    df.loc[df.index[df['file'] == 'nigerian.jsonl'], 'kmeans'] = labels
    '''

    # add cluster lablels for JPL abuse set
    for cat in df['file'].unique():
        if cat not in ['enron.jsonl']:
            nlabels += 1
            df.loc[df.index[df['file'] == cat], 'kmeans'] = nlabels
    return df

def parse_weekly_timestamps(frame):
    '''
    add column with weekly timestamps to frame containing columns: days of week, hours, minutes, seconds
    '''
    frame['Weekly Timestamp'] = frame['Day of Week'] * 24 * 60 * 60 + \
                                frame['Hour'] * 60 * 60 + \
                                frame['Minute'] * 60 + \
                                frame['Seconds']
    return frame

# main method for testing preprocessing functions
if __name__ == '__main__':
    '''
    datapath = '/Users/jeffreygleason 1/Desktop/NewKnowledge/Code/ASED/data'
    all_emails = parse_all_emails(datapath)
    print(all_emails.shape)
    np.savez('all_emails_parsed.npz', all_emails = all_emails)
    '''
    data = '../../all_emails_parsed.npz'
    df = cluster_emails_kmeans(data)
    df.to_pickle('../../all_emails_kmeans_clustered.pkl')
