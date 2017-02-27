import numpy as np
import pysam
import random
import gzip
import pdb

from keras.utils.np_utils import to_categorical

DNA_COMPLEMENT = dict([('A','T'),('T','A'),('G','C'),('C','G'),('N','N')])
ONE_HOT = dict([('A',0),('C',1),('G',2),('T',3)])

#make_complement = lambda seq: ''.join([DNA_COMPLEMENT[s] for s in seq])
#make_reverse_complement = lambda seq: ''.join([DNA_COMPLEMENT[s] for s in seq][::-1])

def partition_sites(options):
    """load sites and partition them into training,
    test and validation sets
    """

    train = []
    test = []
    valid = []

    with gzip.open(options.peak_file,'r') as handle:
        header = handle.next().strip().split()
        cellnames = header[3:]
        for line in handle:
            row = line.strip().split()
            mid = (int(row[1])+int(row[2]))/2
            entry = [row[0], mid-options.window_size/2, mid+options.window_size/2]
            entry.extend([int(r) for r in row[3:]])
            if row[0]==options.test_chromosome:
                test.append(entry)
            else:
                if np.random.rand()>0.01:
                    train.append(entry)
                else:
                    valid.append(entry)

    return train, valid, test, cellnames

def onehot(seq):
    """construct a one-hot encoding of a DNA sequence
    """

    N = len(seq)
    arr = np.zeros((1,4,N), dtype='float32')
    [arr[0,ONE_HOT[s]].__setitem__(i,1) 
     for i,s in enumerate(seq) if ONE_HOT.has_key(s)]
    arr[0,:,~np.any(arr[0]==1,0)] = 0.25

    return arr

def map_cellgroup_to_category(filename):

    cellgroup_counts = dict()
    handle = gzip.open(filename,'r')
    for line in handle:
        loc = line.strip().split()
        try:
            cellgroup_counts[''.join(map(str,loc[3:]))] += 1
        except KeyError:
            cellgroup_counts[''.join(map(str,loc[3:]))] = 1
    handle.close()
    
    #keys = cellgroup_counts.keys()
    #for key in keys:
    #    if cellgroup_counts[key]<1000:
    #        del cellgroup_counts[key]

    keys = cellgroup_counts.keys()
    cellgroup_mapping = dict([(key,k) for k,key in enumerate(keys)])

    return cellgroup_mapping

class DataGenerator():

    def __init__(self, locations, genome_track):

        self._genome_track = genome_track
        self.locations = locations
        self.N = len(self.locations)
        self.batch_index = 0
        self.total_batches_seen = 0
        
    def _flow_index(self, batch_size):

        self.batch_index = 0

        while True:
            if self.batch_index == 0:
                random.shuffle(self.locations)

            current_index = (self.batch_index * batch_size) % self.N
            if self.N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = self.N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (self.locations[current_index : current_index + current_batch_size],
                   current_index, current_batch_size)

    def flow(self, batch_size):

        self.flow_generator = self._flow_index(batch_size)
        return self

    def __iter__(self):

        return self

    def __next__(self):

        return self.next()

    def next(self):

        locations, current_index, current_batch_size = next(self.flow_generator)
        X, Y = self._genome_track.encode_sequences(locations)
        return X, Y

class Genome():

    def __init__(self, fasta_filename, prediction_type, cellgroup_mapping=None):

        self._genome_handle = pysam.FastaFile(fasta_filename)
        self.prediction_type = prediction_type

        if cellgroup_mapping is not None:
            self.cellgroup_mapping = cellgroup_mapping
            self.C = len(self.cellgroup_mapping)

    def _get_dna_sequence(self, location):

        return self._genome_handle.fetch(location[0], location[1], location[2]).upper()

    def encode_sequences(self, locations, batch_size=1000):

        X = np.array([onehot(self._get_dna_sequence(loc))
                      for loc in locations]).astype('float32')

        if self.prediction_type=='cellgroup':
            Y = to_categorical(np.array([self.cellgroup_mapping[''.join(map(str,loc[3:]))] 
                                         for loc in locations]), self.C)

        elif self.prediction_type=='celltype':
            Y = np.array([loc[3:] for loc in locations])

        return X, Y

    def close(self):

        self._genome_handle.close()

def selex_pwms(background={'A':0.25, 'T':0.25, 'G':0.25, 'C':0.25}):

    motifs = dict()

    handle = open("pwms/HTSELEX/selexIDs.txt",'r')
    for pline in handle:
        prow = pline.strip().split()
        if 'ENSG' not in prow[0]:
            continue

        sid = prow[2]
        filename = 'pwms/HTSELEX/%s.dat'%sid

        motifs[sid] = []
        counts = {'A':[], 'T':[], 'G':[], 'C':[]}
        motifhandle = open(filename,'r')
        for line in motifhandle:
            row = line.strip().split(':')
            if row[0] in ['A','T','G','C']:
                counts[row[0]] = map(float,row[1].split())
            else:
                continue
        motifhandle.close()

        motifs[sid] = np.array([counts[nuc] for nuc in ['A','C','G','T']])
        motifs[sid] = motifs[sid]/motifs[sid].sum(0)
    handle.close()

    return motifs

def transfac_pwms(background={'A':0.25, 'T':0.25, 'G':0.25, 'C':0.25}):

    motifs = dict()

    handle = open("pwms/TRANSFAC/transfacIDs.txt", 'r')
    handle.next()
    for pline in handle:
        prow = pline.strip().split('\t')
        if 'ENSG' not in prow[0]:
            continue

        sid = prow[2]
        filename = 'pwms/TRANSFAC/%s.dat'%sid

        motifs[sid] = []
        counts = {'A':[], 'T':[], 'G':[], 'C':[]}
        motifhandle = open(filename,'r')
        for line in motifhandle:
            if line[0]=='#':
                continue
            elif line[0] in ['A','T','G','C']:
                order = line.strip().split()
            else:
                [counts[order[i]].append(c) for i,c in enumerate(map(float,line.strip().split()))]
        motifhandle.close()

        motifs[sid] = np.array([counts[nuc] for nuc in ['A','C','G','T']])
        motifs[sid] = motifs[sid]/motifs[sid].sum(0)
    handle.close()

    return motifs
