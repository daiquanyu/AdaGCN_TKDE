from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
from utils import *
from models import GCN

# Define model evaluation function
def evaluate(sess, model, features, y, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict_target(features, y, support, labels, mask, placeholders)
    outs_val = sess.run([model.clf_loss_t, model.micro_f1_t, model.macro_f1_t,
                         model.weighted_f1_t], feed_dict=feed_dict_val)

    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)

def train(FLAGS, X_t, Y_t, support_t, y_train_t, train_mask_t, 
                 X_s, Y_s, support_s, y_train_s, train_mask_s, placeholders):

    # Create model
    model = model_func(placeholders, X_t[2][1], Y_s.shape[0], Y_t.shape[0], logging=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    micro_f1 = []
    macro_f1 = []

    # Train model
    for epoch in range(FLAGS.epochs):
        # Construct feed dictionary
        ###########################
        # learning rate decaying
        ###########################
        if FLAGS.da_method=='WD':
            # naive method
            if (epoch+1)>=500 and (epoch+1)%100==0:
                FLAGS.lr_gen = FLAGS.lr_gen * FLAGS.shrinking
                FLAGS.lr_dis = FLAGS.lr_dis * FLAGS.shrinking

        feed_dict = construct_feed_dict(X_t, Y_t, support_t, y_train_t, train_mask_t, \
                                        X_s, Y_s, support_s, y_train_s, train_mask_s, placeholders, FLAGS.lr_gen, FLAGS.lr_dis)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        if FLAGS.signal==1:
            # domain adaptation
            # only source has labeled nodes
            if FLAGS.da_method=='WD':
                wd_loss, dis_loss_total = [], []
                for _ in range(FLAGS.D_train_step):
                    outs_dis = sess.run([model.wd_d_op, model.wd_loss, model.dis_loss_total], feed_dict=feed_dict)
                    wd_loss.append(outs_dis[1])
                    dis_loss_total.append(outs_dis[2])
                outs_gen = sess.run([model.opt_op_total_s], feed_dict=feed_dict)

        elif FLAGS.signal==2:
            # domain adaptation
            # both source and target have labeled nodes
            if FLAGS.da_method=='WD':
                for _ in range(FLAGS.D_train_step):
                    outs_dis = sess.run([model.wd_d_op, model.wd_loss, model.dis_loss_total], feed_dict=feed_dict)

                outs_gen = sess.run([model.opt_op_total_s_t], feed_dict=feed_dict)

        ##########################################
        # Recording test results after each epoch
        ##########################################
        test_clf_loss_t, test_micf1, test_macf1, test_wf1, test_duration = evaluate(sess, model, X_t, Y_t, support_t, y_test, test_mask, placeholders)
        print("Epoch:{}".format(epoch+1), "signal={}".format(FLAGS.signal), "S-T:{}-{}".format(FLAGS.source, FLAGS.target), 
              "hiddens={}, dropout={}, l2_param={}".format(FLAGS.hiddens_gcn, FLAGS.dropout, FLAGS.l2_param), 
              "cost=", "{:.3f}".format(test_clf_loss_t), "micro_f1={:.3f}".format(test_micf1), 
              "macro_f1={:.3f}".format(test_macf1), "weighted_f1={:.3f}".format(test_wf1))

        micro_f1.append(test_micf1)
        macro_f1.append(test_macf1)

    #####################
    # Testing
    #####################
    test_clf_loss_t, test_micf1, test_macf1, test_wf1, test_duration = evaluate(sess, model, X_t, Y_t, support_t, y_test, test_mask, placeholders)
    print("signal={}".format(FLAGS.signal), "S-T:{}-{}".format(FLAGS.source, FLAGS.target), 
          "hiddens={}, dropout={}, l2_param={}".format(FLAGS.hiddens_gcn, FLAGS.dropout, FLAGS.l2_param), 
          "cost=", "{:.3f}".format(test_clf_loss_t), "micro_f1={:.3f}".format(test_micf1), 
          "macro_f1={:.3f}".format(test_macf1), "weighted_f1={:.3f}".format(test_wf1))

    return test_micf1, test_macf1

##################################################################################################################
# # Set random seed
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('source', 'citationv1', 'Source dataset string.')  # 'dblpv7', 'citationv1', 'acmv9'
flags.DEFINE_string('target', 'dblpv7', 'Target dataset string.')  # 'dblpv7', 'citationv1', 'acmv9'
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('l2_param', 5e-5, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('signal', 4, 'The network to train: 1-with domain adaptation (source_only), 2-with domain adaptation (source and target).')
flags.DEFINE_float('da_param', 1, 'Weight for wassertein loss.')
flags.DEFINE_float('gp_param', 10, 'Weight for penalty loss.')
flags.DEFINE_integer('D_train_step', 10, 'The number of steps for training discriminator.')
flags.DEFINE_float('shrinking', 0.8, 'Initial learning rate for discriminator.')
flags.DEFINE_float('train_rate', 0, 'The ratio of labeled nodes in target networks.')
flags.DEFINE_float('val_rate', 0, 'The ratio of labeled nodes in validation set in target networks.')
flags.DEFINE_string('hiddens_gcn', '1000|100|16', 'Number of units in different hidden layers for gcn.')
flags.DEFINE_string('hiddens_clf', '', 'Number of units in different hidden layers for supervised classifier.')
flags.DEFINE_string('hiddens_dis', '16', 'Number of units in different hidden layers for dicriminator.')
flags.DEFINE_string('da_method', 'WD', 'Domain adaptation method.')
flags.DEFINE_float('lr_gen', 1.5e-3, 'Initial learning rate.')
flags.DEFINE_float('lr_dis', 1.5e-3, 'Initial learning rate for discriminator.')
flags.DEFINE_boolean('with_metrics', True, 'whether computing f1 scores within tensorflow.')
flags.DEFINE_float('source_train_rate', 0.1, 'The ratio of labeled nodes in target networks.')
#-----------------
# IGCN
flags.DEFINE_integer('num_gcn_layers', 1, 'The number of gcn layers in the IGCN model.')
flags.DEFINE_integer('smoothing_steps', 10, 'The setting of k in A^k.')
flags.DEFINE_string('gnn', 'gcn', 'Convolutional methods.') # 'gcn', 'igcn'
#-----------------

############################################################
############################################################
datasets = ['dblpv7', 'citationv1', 'acmv9']
signal = [1]
target_train_rate = [0] # shall be zero for signal=1; shall not be zero (e.g., 0.05) for signal=2.
smoothing_steps = [10]  # setting smoothing steps according to source training rate
lr_gen = 1.5e-3
lr_dis = 1.5e-3
for s in range(len(signal)):
    FLAGS.signal = signal[s]
    ###################################################################################
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            FLAGS.target = datasets[i]
            if i==j:
                continue
            else:
                FLAGS.source = datasets[j]
            final_micro = []
            final_macro = []
            for k, t in enumerate(target_train_rate):
                FLAGS.lr_gen = lr_gen
                FLAGS.lr_dis = lr_dis
                FLAGS.train_rate = t
                FLAGS.smoothing_steps = smoothing_steps[k]
                ####################
                # Load target data
                ####################
                # determining labeled nodes sampling method
                Y_tmp = sio.loadmat('./input/{}.mat'.format(FLAGS.target))['group']
                N_tmp, M_tmp = Y_tmp.shape
                Ntr_s = int(N_tmp*FLAGS.train_rate)
                tr_label_per_class = Ntr_s // M_tmp
                label_node_min = np.min(np.sum(Y_tmp, axis=0))
                if tr_label_per_class>label_node_min:
                    s_type = 'random'
                else:
                    s_type = 'planetoid'
                #------------------------------
                train_ratio = FLAGS.train_rate
                val_ratio = FLAGS.val_rate
                test_ratio = 1 - FLAGS.train_rate - FLAGS.val_rate
                A_t, X_t, Y_t, y_train_t, y_val, y_test, train_mask_t, val_mask, test_mask = load_mat_data('./input/{}.mat'.format(FLAGS.target),
                                                                       train_ratio, val_ratio, test_ratio, s_type=s_type)
                ####################
                ####################
                #############################################
                # determining labeled nodes sampling method
                Y_tmp = sio.loadmat('./input/{}.mat'.format(FLAGS.source))['group']
                N_tmp, M_tmp = Y_tmp.shape
                Ntr_s = int(N_tmp*FLAGS.source_train_rate)
                tr_label_per_class = Ntr_s // M_tmp
                # Load source data
                label_node_min = np.min(np.sum(Y_tmp, axis=0))
                if tr_label_per_class>label_node_min:
                    s_type = 'random'
                else:
                    s_type = 'planetoid'
                ##############################################
                source_train_ratio = FLAGS.source_train_rate
                source_val_ratio = 0
                source_test_ratio = 1 - source_train_ratio - source_val_ratio
                A_s, X_s, Y_s, y_train_s, y_val_s, y_test_s, train_mask_s, val_mask_s, test_mask_s = load_mat_data('./input/{}.mat'.format(FLAGS.source),
                                                   source_train_ratio, source_val_ratio, source_test_ratio, s_type=s_type)
                ####################
                # Some preprocessing
                ####################
                N_t = Y_t.shape[0]
                N_s = Y_s.shape[0]
                X_t = preprocess_features(X_t)
                X_s = preprocess_features(X_s)
                support_t = [preprocess_adj(A_t)]
                support_s = [preprocess_adj(A_s)]
                num_supports = 1
                model_func = GCN
                ####################
                # Define placeholders
                placeholders = {
                    'support_t': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                    'features_t': tf.sparse_placeholder(tf.float32, shape=tf.constant(X_t[2], dtype=tf.int64)),
                    'labels_t': tf.placeholder(tf.float32, shape=(Y_t.shape[0], Y_t.shape[1])),
                    'labels_mask_t': tf.placeholder(tf.bool, shape=(Y_t.shape[0])),
                    'num_features_nonzero_t': tf.placeholder(tf.int32),  # helper variable for sparse dropout
                    'support_s': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                    'features_s': tf.sparse_placeholder(tf.float32, shape=tf.constant(X_s[2], dtype=tf.int64)),
                    'labels_s': tf.placeholder(tf.float32, shape=(Y_s.shape[0], Y_s.shape[1])),
                    'labels_mask_s': tf.placeholder(tf.bool, shape=(Y_s.shape[0])),
                    'num_features_nonzero_s': tf.placeholder(tf.int32),  # helper variable for sparse dropout
                    'dropout': tf.placeholder_with_default(0., shape=()),
                    'lr_dis': tf.placeholder(tf.float32, shape=()),
                    'lr_gen': tf.placeholder(tf.float32, shape=()),
                    'l': tf.placeholder(tf.float32, shape=()),
                    'source_top_k_list': tf.placeholder(tf.int32, shape=(Y_s.shape[0])),  # for multi-label classification
                    'target_top_k_list': tf.placeholder(tf.int32, shape=(Y_t.shape[0]))
                }
                ###############################################################################
                test_micf1, test_macf1 = train(FLAGS, X_t, Y_t, support_t, y_train_t, train_mask_t, \
                                               X_s, Y_s, support_s, y_train_s, train_mask_s, placeholders)

