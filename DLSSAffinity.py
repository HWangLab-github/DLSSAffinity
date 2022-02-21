#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
np.random.seed(123)
import pandas as pd
from math import sqrt, ceil
import h5py
from sklearn.utils import shuffle
import tensorflow as tf
import os.path
import time
timestamp = time.strftime('%Y-%m-%dT%H:%M:%S') 
import matplotlib as mpl
mpl.use('agg')
import seaborn as sns

from tfbio.data import Featurizer, make_grid, rotate
import tfbio.net1


import json
from collections import OrderedDict
import os
import argparse



datasets = ['training', 'validation', 'test']

def input_dir(path):
    """Check if input directory exists and contains all needed files"""
    global datasets
   
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise IOError('Incorrect input_dir specified: no such directory')
    for dataset_name in datasets:
        dataset_path = os.path.join(path, '%s_set.hdf' % dataset_name)
        if not os.path.exists(dataset_path):
            raise IOError('Incorrect input_dir specified:'
                          ' %s set file not found' % dataset_path)
    return path




parser = argparse.ArgumentParser(
    description='Train 3D colnvolutional neural network on affinity data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

io_group = parser.add_argument_group('I/O')

io_group.add_argument('--input_dir', '-i', default='./database/', type=input_dir,
                      help='directory with training, validation and test sets')
io_group.add_argument('--log_dir', '-l', default='./logdir/',
                      help='directory to store tensorboard summaries')
io_group.add_argument('--output_prefix', '-o', default='./output',
                      help='prefix for checkpoints, predictions and plots')
io_group.add_argument('--grid_spacing', '-g', default=1.0, type=float,
                      help='distance between grid points')
io_group.add_argument('--max_dist', '-d', default=10.0, type=float,
                      help='max distance from complex center')

arc_group = parser.add_argument_group('Netwrok architecture')
arc_group.add_argument('--conv_patch', default=5, type=int,
                       help='patch size for convolutional layers')
arc_group.add_argument('--pool_patch', default=2, type=int,
                       help='patch size for pooling layers')
arc_group.add_argument('--conv_channels', metavar='C', default=[64, 128, 256],
                       type=int, nargs='+',
                       help='number of fileters in convolutional layers')
arc_group.add_argument('--dense_sizes', metavar='D', default=[1000, 500, 200],
                       type=int, nargs='+',
                       help='number of neurons in dense layers')

reg_group = parser.add_argument_group('Regularization')
reg_group.add_argument('--keep_prob', dest='kp', default=0.6, type=float,
                       help='keep probability for dropout')
reg_group.add_argument('--l2', dest='lmbda', default=0.001, type=float,
                       help='lambda for weight decay')
reg_group.add_argument('--rotations', metavar='R', default=list(range(1)),
                       type=int, nargs='+',
                       help='rotations to perform')

tr_group = parser.add_argument_group('Training')
tr_group.add_argument('--learning_rate', default=(1e-5)*1.4, type=float,
                      help='learning rate')
tr_group.add_argument('--batch_size', default=1, type=int,
                      help='batch size')
tr_group.add_argument('--num_epochs', default=8, type=int, #importance factor 
                      help='number of epochs')
tr_group.add_argument('--num_checkpoints', dest='to_keep', default=10, type=int,
                      help='number of checkpoints to keep')

parser.add_argument('--seq_window_lengths', type=int, nargs='+', default = [4], ##nargs=3,
                    help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)')
parser.add_argument('--smi_window_lengths', type=int, nargs='+', default = [4],  ##nargs=3,
                    help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 6 8)')
parser.add_argument('--num_windows', type=int, nargs='+', default = [32], 
                    help='Space seperated list of the number of motif filters corresponding to length list. (ex, --num_windows 100 200 100)')
parser.add_argument('--num_hidden', type=int, default=0, #default=[1024, 1024, 512],
                    help='Number of neurons in hidden layer.')
parser.add_argument('--num_classes', type=int, default=0,
                    help='Number of classes (families).')
parser.add_argument('--max_seq_len', type=int, default=1000,
                    help='Length of input sequences.')
parser.add_argument('--max_smi_len', type=int, default=100,
                    help='Length of input sequences.')
parser.add_argument('--sequence_dataset_path', type=str,default='./database/',
                    help='Directory for input data.')

args = parser.parse_args()


CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }
CHARPROTLEN = 25


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64





def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch] 
                
    return X 


def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X

class DataSet(object):
    def __init__(self, fpath, seqlen, smilen, need_shuffle = False):     
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN
        self.charsmiset = CHARISOSMISET
        self.charsmiset_size = CHARISOSMILEN
    

    def parse_data(self, fpath,  with_label=True):
        ligands = json.load(open(fpath+"all_drugs_smiles.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath+"protein_sequences_dict.txt"), object_pairs_hook=OrderedDict)
    
        XD = []
        XT = []
        
        ligand_keys = ligands.keys()
        protein_keys = proteins.keys()
        
        for d in ligands.keys():
            XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))
        for t in proteins.keys():
            XT.append(label_sequence(proteins[t], self.SEQLEN, self.charseqset))
        return XD, XT, ligand_keys, protein_keys


fpath = args.sequence_dataset_path
dataset = DataSet(fpath = args.sequence_dataset_path,
                  seqlen = args.max_seq_len,
                  smilen = args.max_smi_len,
                  need_shuffle = False )

XD, XT, ligand_keys, protein_keys = dataset.parse_data(fpath = args.sequence_dataset_path)


XD = np.asarray(XD)
XT = np.asarray(XT)

XD_dic = dict(zip(ligand_keys,XD))
XT_dic = dict(zip(protein_keys,XT))

drugcount = XD.shape[0]
targetcount = XT.shape[0]


ids = {}
affinity = {}
coords = {}
features = {}


for dictionary in [ids, affinity, coords, features]:
    for dataset_name in datasets:
        dictionary[dataset_name] = []


for dataset_name in datasets:
    dataset_path = os.path.join(args.input_dir, '%s_set.hdf' % dataset_name)
    with h5py.File(dataset_path, 'r') as f:
        for pdb_id in f: 
            dataset = f[pdb_id]
            coords[dataset_name].append(dataset[:, :3])
            features[dataset_name].append(dataset[:, 3:])
            affinity[dataset_name].append(dataset.attrs['affinity'])
            ids[dataset_name].append(pdb_id)
            

    ids[dataset_name] = np.array(ids[dataset_name])
    affinity[dataset_name] = np.reshape(affinity[dataset_name], (-1, 1))


featurizer = Featurizer()

columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}

charges = []
for feature_data in features['training']:
    charges.append(feature_data[..., columns['partialcharge']])

charges = np.concatenate([c.flatten() for c in charges])

m = charges.mean()
std = charges.std()
print('charges: mean=%s, sd=%s' % (m, std))
print('use sd as scaling factor')
#################################################


def get_batch(dataset_name, indices, rotation=0):
    global coords, features, std
    x = []
    XD = []
    XT = []
    
    for i, idx in enumerate(indices):
        coords_idx = rotate(coords[dataset_name][idx], rotation)
        features_idx = features[dataset_name][idx]
        ids_idx = ids[dataset_name][idx]
        
        x.append(make_grid(coords_idx, features_idx,
                 grid_resolution=args.grid_spacing,
                 max_dist=args.max_dist))
        XD.append(XD_dic[ids_idx])
        XT.append(XT_dic[ids_idx])
    
    x = np.vstack(x)
    XD = np.vstack(XD)
    XT = np.vstack(XT)
    x[..., columns['partialcharge']] /= std
    
    return x, XD, XT

ds_sizes = {dataset: len(affinity[dataset]) for dataset in datasets}
print('type(get_batch(''training'', [0]))', type(get_batch('training', [0])))

osize = 1
isize = 21
in_chnls = 19

for set_name, set_size in ds_sizes.items():
    print('%s %s samples' % (set_size, set_name))

num_batches = {dataset: ceil(size / args.batch_size)
               for dataset, size in ds_sizes.items()}
print('num_batches', num_batches)

def batches(set_name):
    """Batch generator, yields slice indices"""
    global num_batches, args, ds_sizes
    for b in range(num_batches[set_name]):
        bi = b * args.batch_size
        bj = (b + 1) * args.batch_size
        if b == num_batches[set_name] - 1:
            bj = ds_sizes[set_name]
        yield bi, bj


graph = tfbio.net1.make_SB_network(isize=isize, in_chnls=in_chnls, osize=osize,
                                  conv_patch=args.conv_patch,
                                  pool_patch=args.pool_patch,
                                  conv_channels=args.conv_channels,
                                  dense_sizes=args.dense_sizes,
                                  lmbda=args.lmbda,
                                  learning_rate=args.learning_rate)


x = graph.get_tensor_by_name('input/structure_x:0')
XD = graph.get_tensor_by_name('input/sequence_xd:0')
XT = graph.get_tensor_by_name('input/sequence_xt:0')
y = graph.get_tensor_by_name('output/prediction:0')
t = graph.get_tensor_by_name('input/affinity:0')
keep_prob = graph.get_tensor_by_name('fully_connected/keep_prob:0')
print('fully_connected/keep_prob:0', keep_prob)
train = graph.get_tensor_by_name('training/train:0')
mse = graph.get_tensor_by_name('training/mse:0')
global_step = graph.get_tensor_by_name('training/global_step:0')

with graph.as_default():
    saver = tf.train.Saver(max_to_keep=args.to_keep)


train_sample = min(args.batch_size, len(features['training']))
val_sample = min(args.batch_size, len(features['validation']))

print('\n---- TRAINING ----\n')

prefix = os.path.abspath(args.output_prefix) + '-' + timestamp
logdir = os.path.join(os.path.abspath(args.log_dir), os.path.split(prefix)[1])

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    resultmse=[]
    for epoch in range(args.num_epochs):
        print('epoch = ', epoch)
        for rotation in args.rotations:
            print('rotation', rotation)
            x_t, y_t = shuffle(range(ds_sizes['training']), affinity['training'])
            
            for bi, bj in batches('training'):
                session.run(train, feed_dict={x: get_batch('training', x_t[bi:bj], rotation)[0],
                                              XD: get_batch('training', x_t[bi:bj], rotation)[1],
                                              XT: get_batch('training', x_t[bi:bj], rotation)[2],
                                              t: y_t[bi:bj], keep_prob: args.kp})
    
        pred_t = np.zeros((ds_sizes['training'], 1))
        mse_t = np.zeros(num_batches['training'])
        

        for b, (bi, bj) in enumerate(batches('training')):
            weight = (bj - bi) / ds_sizes['training']

            pred_t[bi:bj], mse_t[b] = session.run(
                [y, mse],
                feed_dict={x: get_batch('training', x_t[bi:bj])[0],
                           XD: get_batch('training', x_t[bi:bj])[1],
                           XT: get_batch('training', x_t[bi:bj])[2],
                           t: y_t[bi:bj],
                           keep_prob: 1.0}
            )

            mse_t[b] *= weight

        mse_t = mse_t.sum()

        
        mse_v = 0
        for bi, bj in batches('validation'):
            weight = (bj - bi) / ds_sizes['validation']
            mse_v += weight * session.run(
                mse,
                feed_dict={x: get_batch('validation', range(bi, bj))[0],
                           XD: get_batch('validation', range(bi, bj))[1],
                           XT: get_batch('validation', range(bi, bj))[2],
                           t: affinity['validation'][bi:bj],
                           keep_prob: 1.0}
            )


        print('epoch: %s train error: %s, validation error: %s' % (epoch, mse_t, mse_v))
        resultmse.append([epoch, mse_t, mse_v])
        

        err = float('inf')

        if mse_v <= err:
            err = mse_v
            checkpoint = saver.save(session, prefix, global_step=global_step)
            print('checkpoint', checkpoint)


titlename=['epoch', 'train error', 'validation error']
resultmse=pd.DataFrame(columns=titlename, data=resultmse)
resultmse.to_csv('resultmse.csv', encoding='gbk', index=False)


predictions = []
rmse = {}


with tf.Session(graph=graph) as session:
    tf.set_random_seed(123)
 
    saver.restore(session, os.path.abspath(checkpoint))
    saver.save(session, prefix + '-best') 
    
    for dataset in datasets:
        pred = np.zeros((ds_sizes[dataset], 1))
        mse_dataset = 0.0

        for bi, bj in batches(dataset):
            weight = (bj - bi) / ds_sizes[dataset]
            pred[bi:bj], mse_batch = session.run(
                [y, mse],
                feed_dict={x: get_batch(dataset, range(bi, bj))[0],
                           XD: get_batch(dataset, range(bi, bj))[1],
                           XT: get_batch(dataset, range(bi, bj))[2],
                           t: affinity[dataset][bi:bj],
                           keep_prob: 1.0}
            )
            mse_dataset += weight * mse_batch

        predictions.append(pd.DataFrame(data={'pdbid': ids[dataset],
                                              'real': affinity[dataset][:, 0],
                                              'predicted': pred[:, 0],
                                              'set': dataset}))
        rmse[dataset] = sqrt(mse_dataset)


predictions = pd.concat(predictions, ignore_index=True)
predictions.to_csv(prefix + '-predictions.csv', index=False)


sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()
color = {'training': 'b', 'validation': 'g', 'test': 'r'}

for set_name, tab in predictions.groupby('set'):
    grid = sns.jointplot('real', 'predicted', data=tab, color=color[set_name],
                         space=0.0, xlim=(0, 16), ylim=(0, 16),
                         annot_kws={'title': '%s set (rmse=%.3f)'
                                             % (set_name, rmse[set_name])})
    
    image = tfbio.net1.custom_summary_image(grid.fig)
    grid.fig.savefig(prefix + '-%s.pdf' % set_name)

