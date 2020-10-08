import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import ipdb

def load_lstm_vae_csv(skill = None, ax = None): 
    csv_path = "./lstm_vae.csv" 
    csv = pd.read_csv(csv_path)

    skills = [3, 4, 5, 7, 8, 9]
    no = skills.index(skill) 
    
    _name = csv.iloc[no]['algorithm']
    _tpr = csv.iloc[no]['tpr'] # str with \n
    tpr_str = _tpr.replace('\n','') # remove \n 
    tpr = np.fromstring(tpr_str.strip(']['), dtype=float, sep=' ') # str2float

    _fpr = csv.iloc[no]['fpr']            
    fpr_str = _fpr.replace('\n','') # remove \n       
    fpr = np.fromstring(fpr_str.strip(']['), dtype=float, sep=' ') # str2float

    auroc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label = _name + ": AUC = %.3f"%auroc, lw=2)
    
def load_csv(skill = None, ax = None):
    csv_path = "./tpr_fpr_skill_%s.csv"%skill 
    csv = pd.read_csv(csv_path)
    for i in range(csv.shape[0]):
        _name = csv.iloc[i]['algorithm']
        _tpr = csv.iloc[i]['tpr'] # str with \n
        tpr_str = _tpr.replace('\n','') # remove \n 
        tpr = np.fromstring(tpr_str.strip(']['), dtype=float, sep=' ') # str2float
        _fpr = csv.iloc[i]['fpr']            
        fpr_str = _fpr.replace('\n','') # remove \n       
        fpr = np.fromstring(fpr_str.strip(']['), dtype=float, sep=' ') # str2float
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label = _name + ": AUC = %.3f"%auroc, lw=2)
        if i == 2: # add lstm_vae
            load_lstm_vae_csv(skill = skill, ax = ax)
        
    ax.set_title("The ROC curve of skill %s" %skill)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right')
        

if __name__=="__main__":
    skills = [3, 4, 5, 7, 8, 9]
    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=(16,9))
    axarr = np.atleast_1d(axarr).flatten().tolist()
    for no, skill in enumerate(skills):
        load_csv(skill = skill, ax = axarr[no])
    plt.suptitle("The ROC curves of anomaly detection based on six deep-learning models, \n which tested on the modalities [nf, nt, nl, na, sl, sr] and anomaly bias [0, 2] for six movements, respectively." )
    fig.savefig('roc.png', format='png')
    fig.savefig('roc.eps', format='eps')    
    plt.show()
