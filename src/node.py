# -*- coding: utf-8 -*-
"""cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o-EPqBTHVqVEMfvXDficqw8l_veQ7eWp
"""

#useful
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
from typing import Union, Optional
from time import time
import warnings
warnings.filterwarnings('ignore')

# neural nets
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow_addons as tfa
import tensorflow_probability as tfp

@tf.function
def sparsemoid(inputs: tf.Tensor):
    return tf.clip_by_value(0.5 * inputs + 0.5, 0., 1.)

@tf.function
def identity(x: tf.Tensor):
    return x

class ODST(tf.keras.layers.Layer):
    def __init__(self, n_trees: int = 3, depth: int = 4, units: int = 1, threshold_init_beta: float = 1., **kwargs):
        super(ODST, self).__init__()
        self.initialized = False
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
    
    def build(self, input_shape: tf.TensorShape):
        feature_selection_logits_init = tf.zeros_initializer()
        self.feature_selection_logits = tf.Variable(initial_value=feature_selection_logits_init(shape=(input_shape[-1], self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)        
        
        feature_thresholds_init = tf.zeros_initializer()
        self.feature_thresholds = tf.Variable(initial_value=feature_thresholds_init(shape=(self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)
        
        log_temperatures_init = tf.ones_initializer()
        self.log_temperatures = tf.Variable(initial_value=log_temperatures_init(shape=(self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)
        
        indices = tf.keras.backend.arange(0, 2 ** self.depth, 1)
        offsets = 2 ** tf.keras.backend.arange(0, self.depth, 1)
        bin_codes = (tf.reshape(indices, (1, -1)) // tf.reshape(offsets, (-1, 1)) % 2)
        bin_codes_1hot = tf.stack([bin_codes, 1 - bin_codes], axis=-1)
        self.bin_codes_1hot = tf.Variable(initial_value=tf.cast(bin_codes_1hot, 'float32'), 
                                          trainable=False)
        
        response_init = tf.ones_initializer()
        self.response = tf.Variable(initial_value=response_init(shape=(self.n_trees, self.units, 2**self.depth), dtype='float32'), 
                                    trainable=True)
                
    def initialize(self, inputs):        
        feature_values = self.feature_values(inputs)
        
        # intialize feature_thresholds
        percentiles_q = (100 * tfp.distributions.Beta(self.threshold_init_beta, 
                                                      self.threshold_init_beta)
                         .sample([self.n_trees * self.depth]))
        flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, feature_values)
        init_feature_thresholds = tf.linalg.diag_part(tfp.stats.percentile(flattened_feature_values, percentiles_q, axis=0))
        
        self.feature_thresholds.assign(tf.reshape(init_feature_thresholds, self.feature_thresholds.shape))
        
        
        # intialize log_temperatures
        self.log_temperatures.assign(tfp.stats.percentile(tf.math.abs(feature_values - self.feature_thresholds), 50, axis=0))
        
        
        
    def feature_values(self, inputs: tf.Tensor, training: bool = None):
        feature_selectors = tfa.activations.sparsemax(self.feature_selection_logits)
        # ^--[in_features, n_trees, depth]

        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        # ^--[batch_size, n_trees, depth]
        
        return feature_values
        
    def call(self, inputs: tf.Tensor, training: bool = None):
        if not self.initialized:
            self.initialize(inputs)
            self.initialized = True
            
        feature_values = self.feature_values(inputs)
        
        threshold_logits = (feature_values - self.feature_thresholds) * tf.math.exp(-self.log_temperatures)

        threshold_logits = tf.stack([-threshold_logits, threshold_logits], axis=-1)
        # ^--[batch_size, n_trees, depth, 2]

        bins = sparsemoid(threshold_logits)
        # ^--[batch_size, n_trees, depth, 2], approximately binary

        bin_matches = tf.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # ^--[batch_size, n_trees, depth, 2 ** depth]

        response_weights = tf.math.reduce_prod(bin_matches, axis=-2)
        # ^-- [batch_size, n_trees, 2 ** depth]

        response = tf.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, n_trees, units]
        
        return tf.reduce_sum(response, axis=1)
    
class NODE(tf.keras.Model):
    def __init__(self, units: int = 1, n_layers: int = 1, dropout_rate = 0.1, 
                 link: tf.function = tf.identity, n_trees: int = 3, depth: int = 4, 
                 threshold_init_beta: float = 1., feature_column: Optional[tf.keras.layers.DenseFeatures] = None, 
                 **kwargs):
        super(NODE, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
        self.feature_column = feature_column
        self.dropout_rate = dropout_rate
        
        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(identity)
        else:
            self.feature = feature_column
        
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ensemble = [ODST(n_trees = n_trees,
                              depth = depth,
                              units = units,
                              threshold_init_beta = threshold_init_beta) 
                         for _ in range(n_layers)]
        
        self.link = link
        
    def call(self, inputs, training=None):
        X = self.feature(inputs)
        X = self.bn(X, training=training)
        X = self.dropout(X, training=training)
        
        for i, tree in enumerate(self.ensemble):
            H = tree(X)
            X = tf.concat([X, H], axis=1)
            
        return self.link(H)

def create_NODE(n_layers, units, dropout_rate, depth, n_trees, link, learning_rate):
    
    node = NODE(n_layers = n_layers, units = units, dropout_rate = dropout_rate, 
                depth = depth, n_trees = n_trees, link = link)
    
    node.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), 
                 metrics = [tf.keras.metrics.AUC(name = 'auc')], 
                 loss = 'binary_crossentropy')
    
    return node