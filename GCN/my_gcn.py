
import tensorflow as tf
#from utils import *

import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from gcn.layers import GraphConvolution
from gcn.models import Model
from gcn.metrics import masked_accuracy

class MYGCN (Model):
    def __init__(self, placeholders, input_dim, learning_rate=0.1, num_hidden1=20, num_hidden2=40, pos_loss_multiplier=1, weight_decay=5e-4, **kwargs):
        super(MYGCN, self).__init__()

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

        # training, prediction and loss functions
        self.build()

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.num_hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging)
        )
        self.layers.append(GraphConvolution(input_dim=self.num_hidden1,
                                            output_dim=self.num_hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging)
        )
        self.layers.append(GraphConvolution(input_dim=self.num_hidden2,
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

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs,
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

    def finish_after_training(self, saver, sess, accuracies_test, accuracies_train, cross_entropy_test, predictions, node_names):
        if not os.path.isdir('training'):
            os.mkdir('training')
            print ("Created Training Subdir")
        date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.mkdir('training/' + date_string)
        save_path = 'training/' + date_string + '/'
        print ("Save model to {}".format(save_path + 'model.ckpt'))
        path = saver.save(sess, save_path + 'model.ckpt')

        # save accuracies during training
        with open(save_path + 'accuracies.pkl', 'wb') as f:
            pickle.dump((accuracies_test, accuracies_train, cross_entropy_test), f, pickle.HIGHEST_PROTOCOL)

        # save the predictions
        print (predictions.shape)
        with open(save_path + 'predictions.tsv', 'w') as f:
            f.write('ID\tName\tProb_pos\tProb_neg\n')
            for pred_idx in range(predictions.shape[0]):
                f.write('{}\t{}\t{}\t{}\n'.format(node_names[pred_idx, 0],
                                                  node_names[pred_idx, 1],
                                                  predictions[pred_idx,0],
                                                  predictions[pred_idx,1])
                        )
        # plotting
        print ("Plotting...")
        fig = plt.figure(figsize=(14,8))
        plt.plot(accuracies_test, color='green', label='Accuracy on the test set')
        plt.plot(accuracies_train, color='red', label='Accuracy on the training set')
        plt.legend(loc="lower right")
        fig.savefig(save_path + 'plot.png', dpi=400)
