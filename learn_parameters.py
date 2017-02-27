
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
                        default="cellgroup",
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
        options.model_prefix = options.tag+".model"

    return options

def compute_test_accuracy(X_test, Y_test, model, prediction_type):

    prediction = model.predict(X_test)
    auc = []

    if prediction_type=="cellgroup":

        raise NotImplementedError

    elif prediction_type=="celltype":

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

if __name__=="__main__":

    options = parse_args()

    training, validation, test, cellnames = load.partition_sites(options)

    print "peak file: %s"%options.peak_file
    print "test chromosome: %s"%options.test_chromosome
    print "number of training sites: %d"%len(training)
    print "number of testing sites: %d"%len(test)
    print "number of validation sites: %d"%len(validation)
    print "prediction type: %s"%options.prediction_type

    if options.prediction_type=="cellgroup":
        print "identifying cell groups from observed open chromatin activity ..."
        cellgroup_mappings = load.map_cellgroup_to_category(options.peak_file)
    else:
        cellgroup_mappings = None

    # load reference genome track
    genome_track = load.Genome(options.genome, options.prediction_type, cellgroup_mappings)
    
    # training data generator
    print "setting up a generator for training data ..."
    train_data_generator = load.DataGenerator(training, genome_track)
    train_flow = train_data_generator.flow(batch_size=100)

    # validation data
    print "loading up validation data ..."
    validation_data_generator = load.DataGenerator(validation, genome_track)
    valid_flow = validation_data_generator.flow(batch_size=len(validation))
    X_validation, Y_validation = valid_flow.next()
    N_outputs = Y_validation.shape[1]

    # construct model
    print "building the OrbWeaver model ..."
    if options.prediction_type=='celltype':
        output_activation = 'sigmoid'
        loss = 'binary_crossentropy'
    elif options.prediction_type=='cellgroup':
        output_activation = 'softmax'
        loss = 'categorical_crossentropy'

    network, tfs = model.build_neural_network(N_outputs, output_activation, options.pwms, options.window_size) 

    # set optimization parameters
    print "compiling the OrbWeaver model ..."
    network.compile(optimizer=Adadelta(), 
                    loss=loss,
                    metrics=['accuracy'])

    # callbacks
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50)
    auroc = callbacks.AuROC(options.prediction_type)

    # train model
    print "training the OrbWeaver model ..."
    history = network.fit_generator(train_flow, \
                                    samples_per_epoch=10000, \
                                    nb_epoch=options.num_epochs, \
                                    verbose=2, \
                                    validation_data=(X_validation, Y_validation), \
                                    callbacks=[auroc, early_stopping])

    # evaluate test accuracy
    print "evaluating model on test data ..."
    test_data_generator = load.DataGenerator(test, genome_track)
    test_flow = test_data_generator.flow(batch_size=len(test))
    X_test, Y_test = test_flow.next()

    test_auc = compute_test_accuracy(X_test, Y_test, network, options.prediction_type)
    print test_auc

    genome_track.close()

    print "saving the model architecture and parameters ..."
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
    handle.close()

    print "done."
