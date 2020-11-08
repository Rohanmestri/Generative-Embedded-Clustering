from sklearn.cluster import KMeans
from keras.layers import Lambda
from keras.models import Model, load_model
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from scipy import stats

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model import GEC_Model
from z_mapping import Z_Mapping

import numpy as np


'''
#########################################################
 Finetune - The second step in the process, where we force
 the generate based on z-samples as opposed to x-samples.
 Fix the cluster centers by using K-Means. Defines the
 target distribution and minimizes the KL divergence in
 order to refine the clusters. Retrain or load pretrained
 finetuned model. Added functionality to obtain a PCA plot
 and visualize the latent space.
#########################################################
'''



class Finetune():
    def __init__(self,x,y,dataset,loaded_models,pretrain_optimizer = SGD(lr=1, momentum=0.9),pretrain_epochs = 30,batch_size = 256):
       #Define the hyperparameters
       self.x = x
       self.y = y
       self.autoencoder,self.encoder,self.decoder = loaded_models
       self.update = 500
       self.optimizer = "adam"
       self.batch_size = 512
       self.loss = 'mse'

       if(dataset == "mnist"):
           self.theta = 1.5
       if(dataset == "fashion_mnist"):
           self.theta = 1.3

    def target(self):
       #Define the target distribution with predefined theta value 
       w = self.q ** self.theta / self.q.sum(0)   
       self.p = (w.T / w.sum(1)).T

    def prepare_model(self,Nt,alpha=0.1,beta=10):
       #define the new layer (along with the alpha-beta reparameterization) 
       z_mapping = Z_Mapping(Nt,name='z_map')(self.encoder.output)
       reparam_z_mapping = Lambda(lambda x: beta*(x-alpha),name='reparam_z_map')(z_mapping)

       #create the new encoder, attached with the z-mapping layers
       self.new_encoder = Model(inputs=self.autoencoder.input, outputs=reparam_z_mapping)

       #find initial cluster centre estimates
       centers = KMeans(n_clusters=Nt)
       centers.fit_predict(self.new_encoder.predict(self.x))

       #define the model to be finetuned
       self.model = Model(inputs=self.autoencoder.input, outputs=[z_mapping,self.decoder(reparam_z_mapping)])
       self.model.get_layer(name='z_map').set_weights([centers.cluster_centers_])


    def fit(self,n_iter=2000):
       #define optimizer and the dual loss function with the respective weights
       index,index_array = 0,np.arange(self.x.shape[0])
       self.model.compile(optimizer=self.optimizer, loss={'z_map':'kld','decoder':self.loss},loss_weights=[0.1,2])

       #finetuning epcohs
       for ite in range(int(n_iter)):
          if ite % self.update == 0:
                self.q, _ = self.model.predict(self.x, verbose=0)         
                self.target()  

          if (index + 1) * self.batch_size > self.x.shape[0]:
                loss = self.model.train_on_batch(x=self.x[index * self.batch_size::],
                                                 y=[self.p[index * self.batch_size::], self.x[index * self.batch_size::]])
                index = 0
          else:
                loss = self.model.train_on_batch(x=self.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=[self.p[index * self.batch_size:(index + 1) * self.batch_size],
                                                    self.x[index * self.batch_size:(index + 1) * self.batch_size]])
                index += 1

          print("epoch {0}:  {1}".format(ite,loss))

       return self.new_encoder,self.decoder


    def use_pretrained_model(self,Nt,dataset,alpha=0.1,beta=10):
       autoencoder,encoder,decoder = GEC_Model("pretrained",dimensions = [self.x.shape[-1], 500, 500, 2000, 10]).get_model()
       z_mapping = Z_Mapping(Nt,name='z_map')(self.encoder.output)
       reparam_z_mapping = Lambda(lambda x: beta*(x-alpha),name='reparam_z_map')(z_mapping)

       #create the new encoder, attached with the z-mapping layers
       self.new_encoder = Model(inputs=self.autoencoder.input, outputs=reparam_z_mapping)
       self.generator = decoder

       self.new_encoder.load_weights('finetuned_model/{0}/new_encoder.h5'.format(dataset))
       self.generator.load_weights('finetuned_model/{0}/generator.h5'.format(dataset))

       return self.new_encoder,self.generator
       


    def visualize_latent_space(self):
       #Transform the Nt dimension latent space in 2d space for visualization 
       np.random.seed(42)
       pca = PCA(2)
       pca_vectors = pca.fit_transform(self.new_encoder.predict(self.x))

       df = pd.DataFrame()
       df['pca_1'] = pca_vectors[:,0]
       df['pca_2'] = pca_vectors[:,1]
       df["y"] = self.y

       true_classes = len(set(self.y))

       #plot the results
       plt.figure(figsize=(16,10))
       sns.scatterplot(x="pca_1", y="pca_2",hue="y",palette=sns.color_palette("hls",true_classes),data=df,legend="full",alpha=0.9)


       
      
       

       
       
       

       
