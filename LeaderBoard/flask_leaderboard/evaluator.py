import pandas as pd
import numpy as np
import os
from flask_leaderboard import app, db
import os
from flask_leaderboard.models import Team, User, Question


def Evaluate(filename, question_number):

    submitted_resfile = pd.read_csv(filename, sep = ",", header = None)
    question_file = os.path.join(app.config['RES_FOLDER'], f'Question{question_number}.csv')
    actual_resfile = pd.read_csv(question_file, sep = ",", header = None)

    if(len(submitted_resfile.columns) != len(actual_resfile.columns)):
        return -9

    accuracy = 100* (1 - (submitted_resfile[0] - actual_resfile[0]).sum()**2/(actual_resfile[0] + submitted_resfile[0]).sum()**2)
    return accuracy

def evaluate(filepath, q):
    """
    src code From Kishan Rajput:
    Staff Computer Scientist
    Thomas Jefferson National Accelerator Facility VA USA.

    Modified by:
    Karthik Suresh

    """
    #label_filepaths = ["/w/halld-scshelf2101/ksuresh/EIC-EPIC/Final/Question1/Question1_test_label.csv",
    #                   "/w/halld-scshelf2101/ksuresh/EIC-EPIC/Final/Question2/Question2_test_label.csv",
    #                   "/w/halld-scshelf2101/ksuresh/EIC-EPIC/Final/Question3/Question3_test_label.csv"]
    label_fileContent = pd.read_csv(app.config['Q_KEYS'][q])
    label_fileContent = label_fileContent.apply(pd.to_numeric)
    sorted_labels = label_fileContent.sort_values('eventID')
    sorted_eventID = sorted_labels.eventID.values
    labels = np.array(sorted_labels['PID'])


    status = 'OK'
    filetype = filepath.split(".")[-1]
    if filetype.lower() not in ['csv', 'txt']:
        status = "File type not csv or txt"
        return status, -1

    # Handle files with different separators (Allow only comma separated?)
    try:
        content = pd.read_csv(filepath, header=None)
    except:
        status = "File could not be read..."
        return status, -1

    # Check number of columns
    columns = content.columns
    nColumns = len(columns)
    if nColumns != 2:
        status = "Number of Columns not equal to 2. Check example notebook on formatting the result file."
        return status, -1

    # Trim the content
    content = content[columns[:2]]

    if len(content) < labels.shape[0]:
        status = "Not enough predictions provided"
        return status, -1

    content = content[:labels.shape[0]]
    print(content)
    # Convert df to numeric
    try:
        numeric_content = content.apply(pd.to_numeric)
    except:
        status = "Corrupt file"
        return status, -1

    # Check for NaNs
    if numeric_content.isnull().values.any():
        status = "File contains NaN"
        return status, -1
    if(np.sum(sorted_eventID - numeric_content[0].values) !=0):
        status = "Event IDs do not match"
        return status, -1
    sorted_predictions = numeric_content.sort_values(0)
    predictions = sorted_predictions[1]


    frac_correct = np.mean(labels == predictions)
    #check if less than threshold
    threshold = app.config['Q_THRES'][q]
    if(frac_correct < threshold):
        status = f"Sorry, The performance of submission is : {frac_correct*100:.2f} which is less than threshold : {threshold*100:.2f}"
        return status, 0.0
    score = 50.0 + 50.0*(frac_correct - threshold)/(1 - threshold)
    return status, score
