from keras.optimizers import SGD
from keras.models import Model
from model import GEC_Model

import numpy as np


'''
#########################################################
 Pretrain - pretrain the autoencoder to obtain initial
 cluster center estimates. In the pretrained model, we
 use constrained activity regularization and relax this
 when we use the model for finetuning. Hence, we save the
 weights in a local cache, in order to relax the constraint.
#########################################################
'''


class Pretrain():
    # Initializing the hyperparameters for training the model
    def __init__(self,epochs=300,batch_size=256,loss='mse'):
       self.optimizer = SGD(lr=1, momentum=0.9) # We use SGD in the pretraining
       self.epochs = epochs
       self.batch_size = batch_size
       self.loss = loss

    # Load the processed data
    def load_data(self,x):
       self.x = x

    # fit the model and store the weights in a local cache. Retrieve the model and pretrain
    def fit(self):
       autoencoder,encoder,decoder = GEC_Model(utility="training",dimensions = [self.x.shape[-1], 500, 500, 2000, 10]).get_model()
       autoencoder.compile(optimizer=self.optimizer, loss=self.loss)
       autoencoder.fit(self.x, self.x, batch_size=self.batch_size, epochs=self.epochs)

       autoencoder.save_weights('cache/ae_weights.h5')
       encoder.save_weights('cache/e_weights.h5')
       decoder.save_weights('cache/d_weights.h5')

       autoencoder,encoder,decoder = GEC_Model("pretrained",dimensions = [self.x.shape[-1], 500, 500, 2000, 10]).get_model()
       autoencoder.load_weights('cache/ae_weights.h5')
       encoder.load_weights('cache/e_weights.h5')
       decoder.load_weights('cache/d_weights.h5')
       
       return autoencoder,encoder,decoder

    # Provision to use the pretrained model
    def use_pretrained_model(self,dataset):
       _,encoder,decoder = GEC_Model("pretrained",dimensions = [self.x.shape[-1], 500, 500, 2000, 10]).get_model()
       encoder.load_weights('pretrain_weights/{0}/e_weights.h5'.format(dataset))
       decoder.load_weights('pretrain_weights/{0}/d_weights.h5'.format(dataset))

       autoencoder = Model(encoder.input,decoder(encoder.output))
       
       return autoencoder,encoder,decoder
       

       
