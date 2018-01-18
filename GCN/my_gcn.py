
import tensorflow as tf
#from utils import *

from gcn.layers import GraphConvolution
from gcn.models import Model
from gcn.metrics import masked_accuracy
import io
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorboard import summary as summary_lib
bestSplit = lambda x: (round(math.sqrt(x)), math.ceil(x / round(math.sqrt(x))))

class MyGraphConvolution(GraphConvolution):
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

    def gen_plot(self, mat):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.plot(mat[0])
        plt.title("test")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
            if var.startswith('weight'):
                print (var)
                #png = self.gen_plot(self.vars[var])
                #img = tf.image.decode_png(png.getvalue(), channels=4)
                #img = tf.expand_dims(img, 0)
                tf.summary.image("weight_importance", tf.expand_dims(tf.expand_dims(self.vars[var], 0), -1))

class MYGCN (Model):
    def __init__(self, placeholders, input_dim, learning_rate=0.1, num_hidden1=20, num_hidden2=40, pos_loss_multiplier=1, weight_decay=5e-4, **kwargs):
        super(MYGCN, self).__init__(**kwargs)

        # data placeholders
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # model params
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.weight_decay = weight_decay
        self.pos_loss_multiplier = pos_loss_multiplier
        self.aupr_score = 0
        self.auroc_score = 0

        # training, prediction and loss functions
        self.build()

    def _build(self):
        self.layers.append(MyGraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.num_hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging)
        )
        self.layers.append(MyGraphConvolution(input_dim=self.num_hidden1,
                                            output_dim=self.num_hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging)
        )
        self.layers.append(MyGraphConvolution(input_dim=self.num_hidden2,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x:x,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging)
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
        if self.logging:
            tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs,
                                        self.placeholders['labels'],
                                        self.placeholders['labels_mask']
                                        )
        self.aupr_score, update_op_pr = self.masked_auc_score(self.outputs,
                                                              self.placeholders['labels'],
                                                              self.placeholders['labels_mask'],
                                                              curve='PR')
        self.auroc_score, update_op_roc = self.masked_auc_score(self.outputs,
                                                                self.placeholders['labels'],
                                                                self.placeholders['labels_mask'],
                                                                curve='ROC')
        if self.logging:
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('AUPR', update_op_pr)
            tf.summary.scalar('AUROC', update_op_roc)


    def _aupr_score(self):
        self.aupr_score, _ = self.masked_aupr_score(self.outputs,
                                      self.placeholders['labels'],
                                      self.placeholders['labels_mask']
                                      )

    def masked_softmax_cross_entropy_weight(self, scores, labels, mask):
        """Softmax cross-entropy loss with masking and weight for positives."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=labels)
        loss += labels[:,0]*self.pos_loss_multiplier
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_auc_score(self, scores, labels, mask, curve='PR'):
        prediction = tf.nn.softmax(scores)
        aupr, update_op = tf.metrics.auc(labels=labels[:,0],
                                         predictions=prediction[:,0],
                                         weights=mask,
                                         curve=curve
                                         )
        return aupr, update_op

    def predict(self):
        return tf.nn.softmax(self.outputs)

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
        print("Model restored from file: %s" % path)
