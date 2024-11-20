import scipy.io as sio
import os
import numpy as np
from itertools import combinations

#Define directories

def load_preprocessed_data (preprocessed_dir):
    
    # Load preprocessed data
    files = [f for f in os.listdir(preprocessed_dir) if f.endswith("_preprocessed.mat")]

    ## Initiaize data structure
    eeg_data = {}
    labels_map = {1:0 , 2:1 , 3:2}

    for file in files:
        # parse subject ID and label
        subject_id,emotion_label, _ = file.split("_")
        subject_id = int(subject_id)
        emotion_label = labels_map[int(emotion_label)]
        
        # load data
        data = sio.loadmat(os.path.join(preprocessed_dir, file))
        trial_data = data['data']['trial'][0][0]
        
        # Organize into structure
        if subject_id not in eeg_data:
            eeg_data[subject_id] = {}
        if emotion_label not in eeg_data[subject_id]:
            eeg_data[subject_id][emotion_label] = []
        eeg_data[subject_id][emotion_label].append(trial_data)
        
    return eeg_data


def enumerate_pairs(eeg_data):
    ## Enumerate all possible pairs of emotions
    emotions = list(eeg_data.keys())
    return list(combinations(emotions, 2))

