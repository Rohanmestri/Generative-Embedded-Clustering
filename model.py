from keras import backend as K
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization,MaxPooling2D
from keras.layers import Flatten, Reshape, UpSampling2D, Cropping2D, Conv2DTranspose
from keras import regularizers



'''
#########################################################
 Model - Definition of the autoencoder along with encoder
 and decoder definitions. Provisions for using both the
 type of models, for pretraining as well as pretraining.
#########################################################
'''



class GEC_Model:
    # Define the hyperparameters specific to the model. 
    def __init__(self,utility,dimensions,init = VarianceScaling(scale=1./3., mode='fan_in',distribution='uniform')):
        self.dimensions = dimensions
        self.init = init

        self.activation = 'relu'
        self.weight_init = 'glorot_uniform'
        self.f_dropout = 0.15
        self.Nt = 10

        #"training" - for pretraining; "pretrained" - for finetuning
        if(utility == "training"):
            self.phi = 0.00001
        elif(utility == "pretrained"):
            self.phi = 0.0000001

    # returns autoencoder, encoder and decoder
    def get_model(self):
        input_img = Input(shape=(28,28,1))
        x = Conv2D(8,(3,3),activation = self.activation, padding = 'same')(input_img)    
        x = MaxPooling2D((2,2), padding ='same')(x)
    
    
        shape = K.int_shape(x)
        flatten_1 = Flatten()(x)
        x = flatten_1
        x = Dense(shape[1]*shape[2]*shape[3])(x)


        n_stacks = len(self.dimensions) - 1
        for i in range(n_stacks-1):
           x = Dense(self.dimensions[i + 1], activation=self.activation, kernel_initializer=self.weight_init, name='encoder_{0}'.format(i))(x)

        encoded = Dense(self.Nt, kernel_initializer=self.init,name='latent_space',activity_regularizer=regularizers.l1(self.phi))(x) 
              
        for i in range(n_stacks-1, 0, -1):
           if(i==n_stacks-1):
             decode_ip = Input(shape=(self.dimensions[-1],), name='d_input')
             x = Dense(self.dimensions[i], activation=self.activation, kernel_initializer=self.init, name='decoder_{0}'.format(i))(decode_ip)
             x = Dropout(self.f_dropout)(x)
           else:
             x = Dense(self.dimensions[i], activation=self.activation, kernel_initializer=self.init, name='decoder_{0}'.format(i))(x)     

        x = Dense(shape[1]*shape[2]*shape[3])(x)
        x = Reshape((shape[1],shape[2],shape[3]))(x)
        x = Conv2DTranspose(8,(3,3), activation = self.activation, padding = 'same')(x)
        x = UpSampling2D((2,2))(x)
        decoded = Conv2D(1,(3,3), activation = 'sigmoid', padding ='same')(x)

        encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
        decoder = Model(inputs=decode_ip, outputs=decoded, name='decoder')
        AE = Model(inputs=input_img, outputs=decoder(encoded), name='AE')

        return AE,encoder, decoder
        
     
