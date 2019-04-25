
import tensorflow as tf
#from utils import *

from gcn.layers import GraphConvolution, dot
from gcn.models import Model
from gcn.inits import glorot

#from gcn.metrics import masked_accuracy
import io
import matplotlib.pyplot as plt
import math
bestSplit = lambda x: (round(math.sqrt(x)), math.ceil(x / round(math.sqrt(x))))



def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)



class MyGraphConvolution(GraphConvolution):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, sparse_network=True, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.sparse_network = sparse_network

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()


    def make_weights_plot(self, weight_mat):
        """Create a pyplot plot and save to buffer."""
        fig = plt.figure()
        num_rows, num_cols = bestSplit(20)
        for i in range(20):
            #np_points = np.array(weight_mat[:,i])
            print (weight_mat[:,i].get_shape())
            #plt.bar(np.arange(0, int(24)), weight_mat[:,i])
            plt.plot(weight_mat[:,i])
            plt.title(self.name + '_weights_{}'.format(i))
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf


    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
            tensor = self.vars[var]
            if var.startswith('weight'):
                tf.summary.image("weight_importance", tf.expand_dims(tf.expand_dims(self.vars[var], 0), -1))
            with tf.name_scope('stats_{}'.format(var)):
                tf.summary.scalar('mean', tf.reduce_mean(tensor))
                tf.summary.scalar('max', tf.reduce_max(tensor))
                tf.summary.scalar('min', tf.reduce_min(tensor))
                tf.summary.histogram('histogram', tensor)


    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=self.sparse_network)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        
        return self.act(output)

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs


class MYGCN (Model):
    def __init__(self, placeholders, input_dim, learning_rate=0.1,
                 num_hidden_layers=2, hidden_dims=[20, 40], pos_loss_multiplier=1,
                 weight_decay=5e-4, sparse_network=True, **kwargs):
        super(MYGCN, self).__init__(**kwargs)

        # some checks first
        assert (num_hidden_layers == len(hidden_dims))
        #assert (hidden_units[-1] == output_dim)

        # data placeholders
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.sparse_network = sparse_network

        # model params
        self.weight_decay = weight_decay
        self.pos_loss_multiplier = pos_loss_multiplier
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dims = hidden_dims

        # training, prediction and loss functions
        self.build()


    def _build(self):
        # add intermediate layers
        inp_dim = self.input_dim
        for l in range(self.num_hidden_layers):
            sparse_layer = l==0 if self.sparse_network else False
            self.layers.append(MyGraphConvolution(input_dim=inp_dim,
                                                  output_dim=self.hidden_dims[l],
                                                  placeholders=self.placeholders,
                                                  act=tf.nn.relu,
                                                  dropout=True,
                                                  sparse_inputs=sparse_layer,
                                                  name='gclayer_{}'.format(l+1),
                                                  logging=self.logging,
                                                  sparse_network=self.sparse_network)
            )
            inp_dim = self.hidden_dims[l]
        # add last layer
        layer_n = self.num_hidden_layers + 1
        self.layers.append(MyGraphConvolution(input_dim=self.hidden_dims[-1],
                                              output_dim=self.output_dim,
                                              placeholders=self.placeholders,
                                              act=lambda x: x,
                                              dropout=True,
                                              sparse_inputs=False,
                                              name='gclayer_{}'.format(layer_n),
                                              logging=self.logging,
                                              sparse_network=self.sparse_network)
        )

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += self.masked_softmax_cross_entropy_weight(self.outputs,
                                                              self.placeholders['labels'],
                                                              self.placeholders['labels_mask']
                                                              )

    def _accuracy(self):
        pass

    def masked_softmax_cross_entropy_weight(self, scores, labels, mask):
        """Softmax cross-entropy loss with masking and weight for positives."""
        if scores.shape[1] > 1: # softmax activation in last layer, no weights
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores,
                                                           labels=labels)
        else: # two classes, let's do sigmoid and weights
            prediction = tf.nn.sigmoid(scores)
            loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels,
                                                            logits=prediction,
                                                            pos_weight=self.pos_loss_multiplier)
        # mask loss for nodes we don't know
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def get_performance_metrics(self):
        with tf.variable_scope("evaluation"):
            pred = self.predict()
            _, acc = tf.metrics.accuracy(labels=self.placeholders['labels'],
                                         predictions=tf.greater(pred, 0.5),
                                         weights=self.placeholders['labels_mask']
                                         )
            _, auroc = tf.metrics.auc(labels=self.placeholders['labels'],
                                    predictions=pred,
                                    weights=self.placeholders['labels_mask'],
                                    curve='ROC'
                                    )
            _, aupr = tf.metrics.auc(labels=self.placeholders['labels'],
                                    predictions=pred,
                                    weights=self.placeholders['labels_mask'],
                                    curve='PR',
                                    summation_method='careful_interpolation'
                                    )
            if self.logging:
                tf.summary.scalar('LOSS', self.loss)
                tf.summary.scalar('ACC', acc)
                tf.summary.scalar('AUPR', aupr)
                tf.summary.scalar('AUROC',auroc)
        return self.loss, acc, aupr, auroc

    def masked_auc_score(self, scores, labels, mask, curve='PR'):
        if scores.shape[1] > 1:
            prediction = tf.nn.softmax(scores)
        else:
            prediction = tf.nn.sigmoid(scores)
        aupr, update_op = tf.metrics.auc(labels=labels[:,0],
                                         predictions=prediction[:,0],
                                         weights=mask,
                                         curve=curve,
                                         summation_method='careful_interpolation'
                                         )
        return aupr, update_op

    def predict(self):
        if self.outputs.shape[1] > 1:
            return tf.nn.softmax(self.outputs)
        else:
            return tf.nn.sigmoid(self.outputs)

    def save(self, path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, path)
        print("Model saved in file: %s" % save_path)

    def load(self, path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        saver.restore(sess, path)