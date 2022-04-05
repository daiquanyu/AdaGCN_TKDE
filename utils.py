import math
import numpy as np
import pickle as pkl
import networkx as nx
import scipy
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse import csc_matrix, hstack, vstack
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import sys

import tensorflow as tf


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))

    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features_t, Y_t, support_t, labels_t, labels_mask_t, \
                        features_s, Y_s, support_s, labels_s, labels_mask_s, placeholders, lr_gen, lr_dis):
    """Construct feed dictionary."""
    feed_dict = dict()
    # targetn network
    feed_dict.update({placeholders['labels_t']: labels_t})
    feed_dict.update({placeholders['labels_mask_t']: labels_mask_t})
    feed_dict.update({placeholders['features_t']: features_t})
    feed_dict.update({placeholders['support_t'][i]: support_t[i] for i in range(len(support_t))})
    feed_dict.update({placeholders['num_features_nonzero_t']: features_t[1].shape})
    # source network
    feed_dict.update({placeholders['labels_s']: labels_s})
    feed_dict.update({placeholders['labels_mask_s']: labels_mask_s})
    feed_dict.update({placeholders['features_s']: features_s})
    feed_dict.update({placeholders['support_s'][i]: support_s[i] for i in range(len(support_s))})
    feed_dict.update({placeholders['num_features_nonzero_s']: features_s[1].shape})
    #learning rate
    feed_dict.update({placeholders['lr_gen']: lr_gen})
    feed_dict.update({placeholders['lr_dis']: lr_dis})

    feed_dict.update({placeholders['source_top_k_list']: np.array(np.sum(Y_s, 1), dtype=np.int32)})
    feed_dict.update({placeholders['target_top_k_list']: np.array(np.sum(Y_t, 1), dtype=np.int32)})

    return feed_dict


def construct_feed_dict_target(features, y, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels_t']: labels})
    feed_dict.update({placeholders['labels_mask_t']: labels_mask})
    feed_dict.update({placeholders['features_t']: features})
    feed_dict.update({placeholders['support_t'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero_t']: features[1].shape})
    feed_dict.update({placeholders['dropout']: 0.0})
    feed_dict.update({placeholders['target_top_k_list']: np.array(np.sum(y, 1), dtype=np.int32)})
    return feed_dict

def construct_feed_dict_source(features, y, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels_s']: labels})
    feed_dict.update({placeholders['labels_mask_s']: labels_mask})
    feed_dict.update({placeholders['features_s']: features})
    feed_dict.update({placeholders['support_s'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero_s']: features[1].shape})
    feed_dict.update({placeholders['dropout']: 0.0})
    feed_dict.update({placeholders['source_top_k_list']: np.array(np.sum(y, 1), dtype=np.int32)})
    return feed_dict


def data_splits(y, Ntr, Nval, Nts, s_type='planetoid'):

    np.random.seed(123456)

    if s_type=='planetoid':
        if Ntr>0:
            idx_train = []
            tr_label_per_class = Ntr // y.shape[1]

            for i in range(y.shape[1]):
                if i==y.shape[1]-1:
                    tr_label_per_class = Ntr - tr_label_per_class*(y.shape[1]-1)
                idx_tr_class_i = list(np.random.choice(np.where(y[:, i]!=0)[0], tr_label_per_class, replace=False))
                # print('idx_tr_class_i:', idx_tr_class_i)
                idx_train = idx_train + idx_tr_class_i
            idx = list(set(range(y.shape[0]))-set(idx_train))

            if Nval>0:
                idx_val = list(np.random.choice(idx, Nval, replace=False))
                idx_test = list(set(idx)-set(idx_val))
            else:
                idx_val = None
                idx_test = idx

        elif Ntr==0 and Nval>0:
            idx_train = None
            idx = range(y.shape[0])

            idx_val = list(np.random.choice(idx, Nval, replace=False))
            idx_test = list(set(range(y.shape[0]))-set(idx_val))

        elif Ntr==0 and Nval==0:
            idx_train = None
            idx_val = None
            idx_test = range(y.shape[0])

    elif s_type=='random':
        if Ntr>0:
            idx_train = list(np.random.choice(np.array(range(y.shape[0])), Ntr, replace=False))
            y_train_label = np.sum(y[idx_train, :], axis=0)

            while np.where(y_train_label==0)[0].shape[0]>0:
                idx_train = list(np.random.choice(np.array(range(y.shape[0])), Ntr, replace=False))
                y_train_label = np.sum(y[idx_train, :], axis=0)

            idx = list(set(range(y.shape[0]))-set(idx_train))

            if Nval>0:
                idx_val = list(np.random.choice(idx, Nval, replace=False))
                idx_test = list(set(idx)-set(idx_val))
            else:
                idx_val = None
                idx_test = idx

        elif Ntr==0 and Nval>0:
            idx_train = None
            idx = range(y.shape[0])

            idx_val = list(np.random.choice(idx, Nval, replace=False))
            idx_test = list(set(range(y.shape[0]))-set(idx_val))
            
        elif Ntr==0 and Nval==0:
            idx_train = None
            idx_val = None
            idx_test = range(y.shape[0])

    return idx_train, idx_val, idx_test

def get_splits(y, tr_ratio, val_ratio, ts_ratio, flag=True, s_type='planetoid'):
    """
    flag:
    - True : tr_ratio, val_ratio, ts_ratio are ratios
    - False: tr_ratio, val_ratio, ts_ratio are ratios
    """

    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    N = y.shape[0]
    if flag:
        Ntr = int(N*tr_ratio)
        Nval = int(N*val_ratio)
        Nts = N - Ntr - Nval
    else:
        Ntr = tr_ratio
        Nval = val_ratio
        Nts = ts_ratio

    idx_train, idx_val, idx_test = data_splits(y, Ntr, Nval, Nts, s_type=s_type)

    if Ntr==0:
        y_train = np.zeros(y.shape, dtype=np.int32)
        train_mask = np.array(np.zeros(N), dtype=np.bool)
    else:
        y_train = np.zeros(y.shape, dtype=np.int32)
        y_train[idx_train] = y[idx_train]
        train_mask = sample_mask(idx_train, N)

    if Nval==0:
        y_val = np.zeros(y.shape, dtype=np.int32)
        val_mask = np.array(np.zeros(N), dtype=np.bool)
    else:
        y_val = np.zeros(y.shape, dtype=np.int32)
        y_val[idx_val] = y[idx_val]
        val_mask = sample_mask(idx_val, N)

    y_test = np.zeros(y.shape, dtype=np.int32)
    y_test[idx_test] = y[idx_test]
    test_mask = sample_mask(idx_test, N)
    
    return y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_mat_data(file, train_ratio, val_ratio, test_ratio, s_type='planetoid'):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.csc.csc_matrix):
        X = csc_matrix(X)
    train_num, val_num, test_num = int(Y.shape[0]*train_ratio), int(Y.shape[0]*val_ratio), int(Y.shape[0]*test_ratio)
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(Y, train_num, val_num, test_num, flag=False, s_type=s_type)
    # print()

    return A, X, Y, y_train, y_val, y_test, train_mask, val_mask, test_mask


