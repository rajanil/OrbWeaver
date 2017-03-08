
# additional libs
import cPickle
import argparse
import os
import pdb

# import libs for numerical ops
import numpy as np
import scipy.stats as stats

# optimizer for neural net
from keras.optimizers import Adadelta

# custom libs
import callbacks
import model
import load

def parse_args():

    parser = argparse.ArgumentParser(description="OrbWeaver learns "
        "a neural network model that predicts the open chromatin state of "
        "a genomic locus across multiple cell types based on its DNA sequence alone.")

    parser.add_argument("peak_file",
                        action="store",
                        help="name of a gzipped text file containing "
                        " positional information of genomic loci that are active in at least "
                        " one cell type, and their chromatin activity across all cell types. "
                        " columns of the file should be as follows. "
                        " Chromosome Start End CellType1_Activity CellType2_Activity ... ")

    parser.add_argument("--window_size",
                        type=int,
                        default=500,
                        help="length of DNA sequence centered at each genomic locus "
                        "used for making predictions. (default: 500)")

    parser.add_argument("--test_chromosome",
                        type=str,
                        default="chr18",
                        help="chromosome to be held out as test data, "
                        "to evaluate the performance of the final model (default: chr18)")

    parser.add_argument("--model_prefix",
                        type=str,
                        default=None,
                        help="prefix of file name to store the architecture and "
                        "parameters of the neural network")

    parser.add_argument("--log_file",
                        type=str,
                        default=None,
                        help="file name to log output of the software")

    parser.add_argument("--pwms",
                        type=str,
                        default="pwms",
                        help="path to files with position weight matrices, "
                        "one per transcription factor or genomic feature ")

    parser.add_argument("--genome",
                        type=str,
                        default="genome/hg19.fa",
                        help="path to indexed fasta file containing the relevant "
                        "reference genome sequence")

    parser.add_argument("--prediction_type",
                        type=str,
                        default="celltype",
                        help="specify whether the predicted output should be chromatin activity "
                        "in a specific cell type or a group of cell types. groups are restricted "
                        "to subsets of cells in which the chromatin is open in all cells in the "
                        "subset (and closed in all others) at least 1000 genomic loci. "
                        "(default: cellgroup, options: celltype/cellgroup)")

    parser.add_argument("--num_epochs",
                        type=int,
                        default=100,
                        help="each iteration of stochastic gradient descent uses 100 loci to compute the "
                        "gradient, and each epoch (when the validation error is evaluated) consists of "
                        "10000 loci. this parameter specifies the max number of epochs to run the algorithm. "
                        "if the data set has a large number of loci, increase this parameter to include at least "
                        "one pass through the data. ")

    options = parser.parse_args()

    # if no peak file is provided, throw an error
    if options.peak_file is None:
        parser.error("Need to provide a file of chromatin accessibility peaks, "
                     "and accessibility states in different cell types")

    options.tag = options.peak_file[:-7]

    if options.model_prefix is None:
        options.model_prefix = options.tag+".%s.model"%options.prediction_type

    if options.log_file is None:
        options.log_file = options.tag+".%s.log"%options.prediction_type

    return options

def compute_test_accuracy(X_test, Y_test, model, prediction_type, cellgroup_map_array):

    prediction = model.predict(X_test)
    auc = []

    if prediction_type=="cellgroup":

        prediction = np.dot(prediction, cellgroup_map_array)
        Y_test = np.dot(Y_test, cellgroup_map_array)

    mask = ~np.logical_or(Y_test.sum(1)==0, Y_test.sum(1)==Y_test.shape[1])

    for y,pred in zip(Y_test.T,prediction.T):
        pos = np.logical_and(mask, y==1)
        neg = np.logical_and(mask, y==0)
        try:
            U = stats.mannwhitneyu(pred[pos], pred[neg])[0]
            auc.append(1.-U/(np.count_nonzero(pos)*np.count_nonzero(neg)))
        except ValueError:
            auc.append(0.5)

    return auc

class Logger():

    def __init__(self, log_file):

        self.log_file = log_file
        self.handle = open(self.log_file, 'w')
        self.handle.close()

    def log_this(self, text):

        self.handle = open(self.log_file, 'a')
        self.handle.write(text+'\n')
        print text
        self.handle.close()


if __name__=="__main__":

    options = parse_args()

    logger = Logger(options.log_file)
    logger.log_this("peak file: %s"%options.peak_file)
    logger.log_this("test chromosome: %s"%options.test_chromosome)
    logger.log_this("prediction type: %s"%options.prediction_type)

    training, validation, test, cellnames = load.partition_sites(options)

    logger.log_this("number of training sites: %d"%len(training))
    logger.log_this("number of testing sites: %d"%len(test))
    logger.log_this("number of validation sites: %d"%len(validation))

    if options.prediction_type=="cellgroup":
        logger.log_this("identifying cell groups from observed open chromatin activity ...")
        cellgroup_mappings, cellgroup_map_array = load.map_cellgroup_to_category(options.peak_file)
    else:
        cellgroup_mappings = None
        cellgroup_map_array = None

    # load reference genome track
    genome_track = load.Genome(options.genome, options.prediction_type, cellgroup_mappings)
    
    # training data generator
    logger.log_this("setting up a generator for training data ...")
    train_data_generator = load.DataGenerator(training, genome_track)
    train_flow = train_data_generator.flow(batch_size=100)

    # validation data
    logger.log_this("loading validation data ...")
    validation_data_generator = load.DataGenerator(validation, genome_track)
    valid_flow = validation_data_generator.flow(batch_size=len(validation))
    X_validation, Y_validation = valid_flow.next()
    N_outputs = Y_validation.shape[1]

    # construct model
    logger.log_this("building the OrbWeaver model ...")
    if options.prediction_type=='celltype':
        output_activation = 'sigmoid'
        loss = 'binary_crossentropy'
    elif options.prediction_type=='cellgroup':
        output_activation = 'softmax'
        loss = 'categorical_crossentropy'

    network, tfs = model.build_neural_network(N_outputs, output_activation, options.pwms, options.window_size) 

    # set optimization parameters
    logger.log_this("compiling the OrbWeaver model ...")
    network.compile(optimizer=Adadelta(), 
                    loss=loss,
                    metrics=['accuracy'])

    # callbacks
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50)
    auroc = callbacks.AuROC(options.prediction_type, cellgroup_map_array, logger)

    # train model
    logger.log_this("training the OrbWeaver model ...")
    logger.log_this("cell types: %s"%(' '.join(cellnames)))
    history = network.fit_generator(train_flow, \
                                    samples_per_epoch=10000, \
                                    nb_epoch=options.num_epochs, \
                                    verbose=0, \
                                    validation_data=(X_validation, Y_validation), \
                                    callbacks=[auroc, early_stopping])

    # evaluate test accuracy
    logger.log_this("loading test data ...")
    test_data_generator = load.DataGenerator(test, genome_track)
    test_flow = test_data_generator.flow(batch_size=len(test))
    X_test, Y_test = test_flow.next()

    logger.log_this("evaluating model on test data ...")
    test_auc = compute_test_accuracy(X_test, Y_test, network, options.prediction_type, cellgroup_map_array)
    logger.log_this("test auroc: %s"%(' '.join(['%.4f'%v for v in test_auc])))

    genome_track.close()

    logger.log_this("saving the model architecture and parameters ...")
    # save model architecture
    network_arch = network.to_json()
    handle = open("%s.json"%options.model_prefix,'w')
    handle.write(network_arch)
    handle.close()

    # save model parameters
    network.save_weights("%s.h5"%options.model_prefix, overwrite=True)

    # save TFs
    his = history.history
    handle = open("%s.tfs.pkl"%options.model_prefix,'w')
    cPickle.Pickler(handle,protocol=2).dump(tfs)
    cPickle.Pickler(handle,protocol=2).dump(history)
    cPickle.Pickler(handle,protocol=2).dump(test_auc)
    handle.close()

    logger.log_this("done.")
