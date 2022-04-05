from layers import *
from metrics import *


flags = tf.app.flags
FLAGS = flags.FLAGS

def define_variables(hiddens, weight_name, bias_name, flag=False):
    variables = {}
    for i in range(len(hiddens)-1):
        variables[weight_name.format(i)] = glorot([hiddens[i], hiddens[i+1]], name=weight_name.format(i))
        if flag:
            variables[bias_name.format(i)] = zeros([hiddens[i+1]], name=bias_name.format(i))

    return variables

####################

class GCN(object):
    def __init__(self, 
                 placeholders, 
                 input_dim, 
                 N_s, 
                 N_t, 
                 bias_flag = False, 
                 c_type='multi-label', 
                 **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.da_method = FLAGS.da_method
        self.c_type = c_type
        self.bias_flag = bias_flag

        self.inputs_t = placeholders['features_t']
        self.inputs_s = placeholders['features_s']
        self.output_dim = placeholders['labels_t'].get_shape().as_list()[1]

        self.N = N_s if N_s<N_t else N_t
        self.N_s = N_s
        self.N_t = N_t

        self.hiddens_gcn = [input_dim] + [int(h) for h in FLAGS.hiddens_gcn.split('|')]
        self.hiddens_clf = [self.hiddens_gcn[-1]] + [int(h) for h in FLAGS.hiddens_clf.split('|') if h!=''] + [self.output_dim]

        self.hiddens_dis = [self.hiddens_gcn[-1]] + [int(h) for h in FLAGS.hiddens_dis.split('|') if h!=''] + [1]

        self.vars = {}

        self.layers_t = []
        self.activations_t = []
        self.layers_s = []
        self.activations_s = []

        self.clf_outputs_t = None
        self.hiddens_t = None

        self.clf_outputs_s = None
        self.hiddens_s = None

        self.clf_loss = 0
        self.clf_loss_t = 0
        self.clf_loss_s = 0
        self.opt_op = None
        self.opt_op_t = None
        self.opt_op_s = None
        self.placeholders = placeholders
        self.build()

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        theta_C = [v for v in tf.global_variables() if 'clf' in v.name]
        theta_D = [v for v in tf.global_variables() if 'dis' in v.name]
        theta_G = [v for v in tf.global_variables() if 'gcn' in v.name]

        ##############
        # Generator
        ##############
        # Build sequential layer model
        self.activations_t.append(self.inputs_t)
        for layer in self.layers_t:
            hidden = layer(self.activations_t[-1])
            self.activations_t.append(hidden)
        self.hiddens_t = self.activations_t[-1]

        self.activations_s.append(self.inputs_s)
        for layer in self.layers_s:
            hidden = layer(self.activations_s[-1])
            self.activations_s.append(hidden)
        self.hiddens_s = self.activations_s[-1]

        ########################
        # Classifier
        ########################
        self.clf_outputs_t = self._classifier(self.hiddens_t)
        self.clf_outputs_s = self._classifier(self.hiddens_s)

        ########################
        # Discriminator
        ########################
        if self.da_method=='WD':
            # generate samples for gradient penaly term
            #-----------------------------------------------------------------------------------
            if self.N_s < self.N_t:
                hiddens_s = tf.slice(self.hiddens_s, [0, 0], [self.N, -1])
                hiddens_t_1 = tf.slice(self.hiddens_t, [0, 0], [self.N, -1])
                hiddens_t_2 = tf.slice(self.hiddens_t, [self.N_t-self.N-1, 0], [self.N, -1])

                hiddens_s = tf.concat([hiddens_s, hiddens_s], axis=0)
                hiddens_t = tf.concat([hiddens_t_1, hiddens_t_2], axis=0)

                alpha = tf.random_uniform(shape=[2*self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha*difference)
            elif self.N_s > self.N_t:
                hiddens_s_1 = tf.slice(self.hiddens_s, [0, 0], [self.N, -1])
                hiddens_s_2 = tf.slice(self.hiddens_s, [self.N_s-self.N-1, 0], [self.N, -1])
                hiddens_t = tf.slice(self.hiddens_t, [0, 0], [self.N, -1])

                hiddens_s = tf.concat([hiddens_s_1, hiddens_s_2], axis=0)
                hiddens_t = tf.concat([hiddens_t, hiddens_t], axis=0)

                alpha = tf.random_uniform(shape=[2*self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha*difference)
            else:
                hiddens_s = tf.slice(self.hiddens_s, [0, 0], [self.N, -1])
                hiddens_t = tf.slice(self.hiddens_t, [0, 0], [self.N, -1])
                alpha = tf.random_uniform(shape=[self.N, 1], minval=0., maxval=1.)
                difference = hiddens_s - hiddens_t
                interpolates = hiddens_t + (alpha*difference)
            #-----------------------------------------------------------------------------------

            hiddens_whole = tf.concat([self.hiddens_s, self.hiddens_t, interpolates], axis=0)
            
            # critic loss
            critic_out = self._discriminator(hiddens_whole)
            critic_s = tf.slice(critic_out, [0, 0], [self.N_s, -1])
            critic_t = tf.slice(critic_out, [self.N_s, 0], [self.N_t, -1])
            self.wd_loss = (tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t))

            # gradient penalty
            gradients = tf.gradients(critic_out, [hiddens_whole])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            self.gradient_penalty = tf.reduce_mean((slopes-1.)**2)

            # optimizer
            self.dis_loss_total = -self.wd_loss+FLAGS.gp_param*self.gradient_penalty
            self.wd_d_op = tf.train.AdamOptimizer(self.placeholders['lr_dis']).minimize(self.dis_loss_total, var_list=theta_D)
        else:
            raise RuntimeError('Wrong DA Type!')

        ####################
        # Weight decay loss
        ####################
        self.l2_loss = FLAGS.l2_param * tf.add_n([tf.nn.l2_loss(v) for v in self.vars.values() if ('bias' not in v.name and 'dis' not in v.name)])

        ###################
        # supervised loss
        ###################
        if self.c_type == 'multi-label':
            self.clf_loss_t = self.l2_loss + masked_sigmoid_cross_entropy(self.clf_outputs_t, self.placeholders['labels_t'],
                                                              self.placeholders['labels_mask_t'])
            self.clf_loss_s = self.l2_loss + masked_sigmoid_cross_entropy(self.clf_outputs_s, self.placeholders['labels_s'],
                                                              self.placeholders['labels_mask_s'])
            self.clf_loss = self.clf_loss_t + self.clf_loss_s - self.l2_loss
            self.clf_loss_pure = self.clf_loss - self.l2_loss

        ###################
        # total loss
        ###################
        if self.da_method=='WD':
            self.total_loss_s = self.clf_loss_s + FLAGS.da_param*self.wd_loss
            self.total_loss_s_t = self.clf_loss + FLAGS.da_param*self.wd_loss
        else:
            raise RuntimeError('Wrong DA Type!')

        self.optimizer = tf.train.AdamOptimizer(self.placeholders['lr_gen'])
        self.opt_op_total_s = self.optimizer.minimize(self.total_loss_s, var_list=theta_G+theta_C)  # supervised loss and domain adaption loss
        self.opt_op_total_s_t = self.optimizer.minimize(self.total_loss_s_t, var_list=theta_G+theta_C)  # supervised loss and domain adaption loss

        # Build metrics
        if FLAGS.with_metrics:
            self._accuracy()

    def _classifier(self, input_tensor, act=tf.nn.relu):
        """classification"""
        hiddens = input_tensor
        for i in range(len(self.hiddens_clf)-2):
            hiddens = act(tf.matmul(hiddens, self.vars['clf_{}_weights'.format(i)])+self.vars['clf_{}_bias'.format(i)])
        layer = len(self.hiddens_clf)-2
        outputs = tf.matmul(hiddens, self.vars['clf_{}_weights'.format(layer)]) + self.vars['clf_{}_bias'.format(layer)]

        return outputs

    def _discriminator(self, input_tensor, act=tf.nn.tanh):
        """discriminator"""
        hiddens = input_tensor
        for i in range(len(self.hiddens_dis)-2):
            hiddens = act(tf.matmul(hiddens, self.vars['dis_{}_weights'.format(i)])+self.vars['dis_{}_bias'.format(i)])
        layer = len(self.hiddens_dis)-2
        outputs = tf.matmul(hiddens, self.vars['dis_{}_weights'.format(layer)]) + self.vars['dis_{}_bias'.format(layer)]

        return outputs

    def _build(self):
        self._create_variables()
        if FLAGS.gnn=='gcn':
            self._create_generator()
        elif FLAGS.gnn=='igcn':
            self._create_generator_igcn()

    def _create_variables(self):
        # define variables
        vars_gcn = define_variables(self.hiddens_gcn, weight_name='gcn_{}_weights', bias_name = 'gcn_{}_bias', flag=self.bias_flag)
        vars_clf = define_variables(self.hiddens_clf, weight_name='clf_{}_weights', bias_name = 'clf_{}_bias', flag=True)
        vars_dis = define_variables(self.hiddens_dis, weight_name='dis_{}_weights', bias_name = 'dis_{}_bias', flag=True)

        self.vars = dict(vars_gcn)
        self.vars.update(vars_clf)
        self.vars.update(vars_dis)

    def _create_generator(self):
        # define model layers for generator
        for i in range(len(self.hiddens_gcn)-1):
            if i==0: 
                sparse = True
                act = tf.nn.relu
            else: 
                sparse = False
                act = tf.nn.relu

            if self.bias_flag:
                bias = self.vars['gcn_{}_bias'.format(i)]
            else:
                bias = None

            self.layers_t.append(GraphConvolution(input_dim=self.hiddens_gcn[i],
                                                  output_dim=self.hiddens_gcn[i+1],
                                                  placeholder_dropout = self.placeholders['dropout'],
                                                  placeholder_support = self.placeholders['support_t'],
                                                  placeholder_num_features_nonzero = self.placeholders['num_features_nonzero_t'],
                                                  weights = self.vars['gcn_{}_weights'.format(i)],
                                                  bias = bias,
                                                  act=act,
                                                  dropout=True,
                                                  sparse_inputs=sparse,
                                                  logging=self.logging))

            self.layers_s.append(GraphConvolution(input_dim=self.hiddens_gcn[i],
                                                  output_dim=self.hiddens_gcn[i+1],
                                                  placeholder_dropout = self.placeholders['dropout'],
                                                  placeholder_support = self.placeholders['support_s'],
                                                  placeholder_num_features_nonzero = self.placeholders['num_features_nonzero_s'],
                                                  weights = self.vars['gcn_{}_weights'.format(i)],
                                                  bias = bias,
                                                  act=act,
                                                  dropout=True,
                                                  sparse_inputs=sparse,
                                                  logging=self.logging))

    def _create_generator_igcn(self):
        # reference: label efficient semi-supervised learning via graph filtering
        # define model layers for generator
        for i in range(len(self.hiddens_gcn)-1):
            if i==0: 
                sparse = True
                act = tf.nn.relu
            else: 
                sparse = False
                act = tf.nn.relu

            if self.bias_flag:
                bias = self.vars['gcn_{}_bias'.format(i)]
            else:
                bias = None

            if i<FLAGS.num_gcn_layers:
                self.layers_t.append(GraphConvolution(input_dim=self.hiddens_gcn[i],
                                                      output_dim=self.hiddens_gcn[i+1],
                                                      placeholder_dropout = self.placeholders['dropout'],
                                                      placeholder_support = self.placeholders['support_t'],
                                                      placeholder_num_features_nonzero = self.placeholders['num_features_nonzero_t'],
                                                      weights = self.vars['gcn_{}_weights'.format(i)],
                                                      bias = bias,
                                                      act=act,
                                                      dropout=True,
                                                      sparse_inputs=sparse,
                                                      logging=self.logging))

                self.layers_s.append(GraphConvolution(input_dim=self.hiddens_gcn[i],
                                                      output_dim=self.hiddens_gcn[i+1],
                                                      placeholder_dropout = self.placeholders['dropout'],
                                                      placeholder_support = self.placeholders['support_s'],
                                                      placeholder_num_features_nonzero = self.placeholders['num_features_nonzero_s'],
                                                      weights = self.vars['gcn_{}_weights'.format(i)],
                                                      bias = bias,
                                                      act=act,
                                                      dropout=True,
                                                      sparse_inputs=sparse,
                                                      logging=self.logging))
            else:
                self.layers_t.append(Dense(placeholder_dropout = self.placeholders['dropout'],
                                           placeholder_num_features_nonzero = self.placeholders['num_features_nonzero_t'],
                                           weights = self.vars['gcn_{}_weights'.format(i)],
                                           bias = bias,
                                           act=act,
                                           dropout=True,
                                           sparse_inputs=sparse,
                                           logging=self.logging))

                self.layers_s.append(Dense(placeholder_dropout = self.placeholders['dropout'],
                                           placeholder_num_features_nonzero = self.placeholders['num_features_nonzero_s'],
                                           weights = self.vars['gcn_{}_weights'.format(i)],
                                           bias = bias,
                                           act=act,
                                           dropout=True,
                                           sparse_inputs=sparse,
                                           logging=self.logging))

    def _accuracy(self):
        if self.c_type == 'multi-label':
            # target
            predictions_t = multi_label_hot(tf.sigmoid(self.clf_outputs_t))
            self.micro_f1_t, self.macro_f1_t, self.weighted_f1_t, self.TP_t, self.FP_t, self.FN_t = f1_score(predictions_t,
                             self.placeholders['labels_t'], self.placeholders['labels_mask_t'])
            # source
            predictions_s = multi_label_hot(tf.sigmoid(self.clf_outputs_s))
            self.micro_f1_s, self.macro_f1_s, self.weighted_f1_s, self.TP_s, self.FP_s, self.FN_s = f1_score(predictions_s,
                             self.placeholders['labels_s'], self.placeholders['labels_mask_s'])
