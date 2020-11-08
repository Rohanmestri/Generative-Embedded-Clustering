from keras.layers import Layer,InputSpec, Lambda, BatchNormalization
from keras import backend as K


'''
#########################################################
 Z_Mapping - The definition of the layer which maps the
 x-space datapoints to the z-space. The weights of this
 layer are a t-distribution similarity estimation with respect
 to each cluster centre. This implementation is similar
 to X.Guo et al.'s implementation. 
#########################################################
'''


class Z_Mapping(Layer):
    def __init__(self, Nt, weights=None, alpha=1, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Z_Mapping, self).__init__(**kwargs)
        
        self.alpha = alpha
        self.Nt = Nt
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.centers = self.add_weight(shape=(self.Nt, input_dim), initializer='glorot_uniform', name='centers')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    # This function measures the t-distribution similarity.
    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.centers), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.Nt

    def get_config(self):
        config = {'Nt': self.Nt}
        base_config = super(Z_Mapping, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
