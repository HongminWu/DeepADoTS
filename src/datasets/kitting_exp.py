import numpy as np
import pandas as pd

from .real_datasets import RealDataset
import ipdb

class KittingExp(RealDataset):
    def __init__(self, seed, skill):
        super().__init__(
            name="Kitting Experiment", raw_path='none', file_name='kitting_exp_skill_%s.npz'%skill
        )
        self.seed = seed

    def load(self):
        (a, b), (c, d) = self.get_data_dagmm()
        self._data = (a, b, c, d)

    def get_data_dagmm(self):
        """
        return: (X_train, y_train), (X_test, y_test)
        """
        data = np.load(self.processed_path)
        np.random.seed(self.seed)
        
        labels = data['kitting_skill'][:, -1]
        features = data['kitting_skill'][:, :-1]

        N, D = features.shape 
        normal_data = features[labels==0] # normal is lablled as 0 
        normal_labels = labels[labels==0]


        anomaly_data = features[labels==1] # anomaly is lablled as 1
        anomaly_labels = labels[labels==1]

        N_normal = normal_data.shape[0]
        
        randIdx = np.arange(N_normal) # shuffle the data
        np.random.shuffle(randIdx)
        N_train = N_normal *2 // 3  # 2/3 for training and 1/3 for testing
        train = normal_data[randIdx[:N_train]]            # train data
        train_labels = normal_labels[randIdx[:N_train]]

        test = normal_data[randIdx[N_train:]]             # test data
        test_labels = normal_labels[randIdx[N_train:]]
        test = np.concatenate((test, anomaly_data),axis=0) # concatenate the remaining normal and anomaly as consequent testing data
        test_labels = np.concatenate((test_labels, anomaly_labels),axis=0)

        print 
        print("shape of train:{}, shape of test:{}".format(train.shape, test.shape))
        print
        print("shape of normal_data:{}, shape of anomaly_data:{}".format(normal_data.shape, anomaly_data.shape))
        print
        input("press Enter to continue!")
        return (pd.DataFrame(data=train), pd.DataFrame(data=train_labels)), (
            pd.DataFrame(data=test), pd.DataFrame(data=test_labels))
    
    
