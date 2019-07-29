from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np

def anomaly_classification_percentile(anomaly_scores, percentile):
    '''
        Classify all anomaly scores above a certain percentile as anomalous
    '''
    return np.array((anomaly_scores > np.percentile(anomaly_scores, percentile)).astype(int))

def anomaly_classification_zscore(anomaly_scores, zscore):
    '''
        Classify all anomaly scores above a certain z-score as anomalous
    '''
    zscores = stats.zscore(anomaly_scores)
    return (zscores > zscore).astype(int)

def evaluate(y_truth, y_pred, target_names = ['non-anomalous', 'anomalous']):
    '''
        Evaluation utility prints out the sklearn classification report
        and accuracy score for the comparison between y_truth and y_pred
    '''
    print('Evaluating on {} predictions'.format(len(y_pred)))
    print("Accuracy = ", accuracy_score(y_truth, y_pred))
    print(classification_report(y_truth, y_pred, target_names=target_names))
