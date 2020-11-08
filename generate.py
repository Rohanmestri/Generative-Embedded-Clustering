from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt 
import numpy as np


'''
#########################################################
 Generate - Loads the finetuned model and makes generates
 the z-space datapoints using the newly trained encoder.
 Based on these datapoints, we estimate the distribution
 of the Gaussian Components in this dataspace. Samples
 from these distributions are fed into the generator.
#########################################################
'''



class Generate():
    def __init__(self,x,Nc,loaded_models):
       #initialize the hyperparameters and load models
       self.x = x
       self.Nc = Nc
       self.new_encoder,self.generator = loaded_models

    def learn_distributions(self):
       #Initialize the GMM object with the prior number of components Nc and learn parameters
       print("------------ Running the Parameter Estimation --------------------")
       gmm = GaussianMixture(n_components=self.Nc, covariance_type='full',verbose=1).fit(self.new_encoder.predict(self.x))
       self.means = gmm.means_
       self.covs = gmm.covariances_
       return self.means,self.covs

    def generate_samples_from_component(self,comp_no):
       #generate a nxn grid with randomly generated samples from a cluster distribution
       n,d = 15,28
       figure = np.zeros((d*n,d*n))
       mean,cov = self.means[comp_no],self.covs[comp_no]
       test = np.random.multivariate_normal(mean,cov,225)

       i,j=0,0
       for sample in test:   
          sample = np.array([sample])
          z_decoded = self.generator.predict(sample)
          z_decoded = z_decoded.reshape(d,d)
          figure[i*d:(i + 1)*d,j*d:(j + 1)*d] = z_decoded

          j+=1
          if(j==15):
            i+=1
            j=0

       #plot the samples
       plt.figure(figsize=(10, 10))
       plt.imshow(figure)
       plt.show()
       
