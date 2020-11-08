from keras.datasets import mnist,fashion_mnist
from pretrain import Pretrain
from finetune import Finetune
from generate import Generate

import numpy as np

'''
#########################################################
 GEC - Controls the operations of the three step procedure
 as mentioned in the paper. Sets hyperparameters like Nt
 (latent space dimensions), Nc (Number of Gaussian Components)
 and the dataset that will be used.
#########################################################
'''



class GEC():
    def __init__(self,dataset):
        #initialize the hyperparameters
        self.Nc = 10
        self.Nt = 10
        self.dataset = dataset


    def load_dataset(self):
        #load either the "mnist" or the "fashion_mnist" dataset
        print("Initialising the {0} dataset".format(self.dataset))
        
        if(self.dataset == "mnist"):
          (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif(self.dataset == "fashion_mnist"):
          (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        #We use the entire dataset and preprocess the dataset
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = np.divide(x, 255.)
        x = x.reshape((70000,28,28,1))

        self.x = x
        self.y = y

    def pretrain(self,retrain=False):
        #If retrain is False, then load the pretrained model. Otherwise, train from scratch
        pretrained = Pretrain()
        pretrained.load_data(self.x)
        if(retrain):
            self.autoencoder,self.encoder,self.decoder = pretrained.fit()  
        else:
            self.autoencoder,self.encoder,self.decoder = pretrained.use_pretrained_model(self.dataset)
            
    def finetune(self,retrain=False):
        #If retrain is False, then load the pretrained model. Otherwise, train from scratch
        models = [self.autoencoder,self.encoder,self.decoder]
        ft = Finetune(self.x,self.y,self.dataset,models)
        if(retrain):
            ft.prepare_model(self.Nt)
            self.new_encoder,self.generator = ft.fit()

            # visualizing the latent space by a PCA Decomposition into 2 dimensions.
            ft.visualize_latent_space()
        else:
            self.new_encoder,self.generator = ft.use_pretrained_model(self.Nt,self.dataset)
            ft.visualize_latent_space()

    def generate(self):
        #creating an object to get the parametric estimates of the clustered distributions 
        models = [self.new_encoder,self.generator]
        gen = Generate(self.x,self.Nc,models)
        gen.learn_distributions()

        #The argument below ranges from 0 <= n <Nc
        gen.generate_samples_from_component(3)



#Driver function to use the above class                       
if __name__ == '__main__':
   #modify parameters in this block
   g = GEC("mnist")           #change dataset 
   g.load_dataset()            
   g.pretrain(retrain=False)  #Toggle to use pretrained model/retrain model from scratch
   g.finetune(retrain=False)   #Toggle to use pretrained model/retrain model from scratch
   g.generate()


