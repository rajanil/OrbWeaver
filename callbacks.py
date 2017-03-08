import numpy as np
import scipy.stats as stats
import time
import pdb

from keras.callbacks import Callback, EarlyStopping

class AuROC(Callback):

    def __init__(self, predtype, map_array, logger):

        self.prediction_type = predtype
        self.map_array = map_array
        self.logger = logger
        self.current_time = time.time()

    def on_train_begin(self, logs={}):
        self.values = []
        self.current_time = time.time()

    def on_epoch_end(self, epoch, logs={}):

        values = []
        prediction = self.model.predict(self.model.validation_data[0])
        Y = self.model.validation_data[1][0]

        if self.prediction_type=="cellgroup":

            prediction = np.dot(prediction, self.map_array)
            Y = np.dot(Y, self.map_array)

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
        epoch_time = time.time()-self.current_time
        self.logger.log_this("epoch_%d_auroc: %s (%ds)"%(epoch, ' '.join(['%.4f'%v for v in self.values[-1]]), int(epoch_time)))
        self.current_time = time.time()
