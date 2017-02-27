import numpy as np
import scipy.stats as stats
import pdb

from keras.callbacks import Callback, EarlyStopping

class AuROC(Callback):

    def __init__(self, predtype):

        self.prediction_type = predtype

    def on_train_begin(self, logs={}):
        self.values = []

    def on_epoch_end(self, epoch, logs={}):

        values = []
        prediction = self.model.predict(self.model.validation_data[0])
        Y = self.model.validation_data[1][0]

        if self.prediction_type=="cellgroup":

            # mask out non-variable windows
            mask = ~np.logical_or(Y[:,0]==1, Y[:,-1]==1)

            # compute AUC for each celltype
            ipsc_idx = [4,5,6,7]
            lcl_idx = [2,3,6,7]
            cmyo_idx = [1,3,5,7]
            for idx in [ipsc_idx, lcl_idx, cmyo_idx]:
                y = np.sum(Y[:,idx],1)
                pred = np.sum(prediction[:,idx], 1)
                pos = np.logical_and(mask, y==1)
                neg = np.logical_and(mask, y==0)
                try:
                    U = stats.mannwhitneyu(pred[pos], pred[neg])[0]
                    values.append(1.-U/(np.count_nonzero(pos)*np.count_nonzero(neg)))
                except ValueError:
                    values.append(0.5)

            # compute AUC for cell-specificity
            idx = [4,2,1]
            for i in idx:
                y = Y[:,i]==1
                pred = prediction[:,i]
                pos = np.logical_and(mask,y==1)
                neg = np.logical_and(mask,y==0)
                try:
                    U = stats.mannwhitneyu(pred[pos], pred[neg])[0]
                    values.append(1.-U/(np.count_nonzero(pos)*np.count_nonzero(neg)))
                except ValueError:
                    values.append(0.5)

        elif self.prediction_type=="celltype":

            mask = ~np.logical_or(Y.sum(1)==0, Y.sum(1)==Y.shape[1])

            for y,pred in zip(Y.T,prediction.T):
                pos = np.logical_and(mask, y==1)
                neg = np.logical_and(mask, y==0)
                try:
                    U = stats.mannwhitneyu(pred[pos], pred[neg])[0]
                    values.append(1.-U/(np.count_nonzero(pos)*np.count_nonzero(neg)))
                except ValueError:
                    values.append(0.5)

        self.values.append(values)
        print "auroc:", self.values[-1]


