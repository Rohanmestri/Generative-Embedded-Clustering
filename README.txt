############################################ README ##########################################

I. Dependencies

1. Tensorflow-GPU 1.12.0
2. Keras 2.1.6
3. Numpy 1.14.3
4. Scikit-Learn 0.23.1
5. Scipy 1.1.0
6. Seaborn 0.9.0
7. Pandas 0.23.0
8. Matplotlib 2.2.2


II. How to Run 

1. Import the entire directory into the Environment with the above-mentioned dependencies
2. Run GEC.py for the execution of the entire 3-step procedure.
3. Provisions to retrain a model from scratch in either the pretraining or the finetuning
   stage is present in GEC.py (Toggle Parameter --> Retrain)
4. Specific Hyperparameters can be easily found in GEC.py. If not found in this file, the 
   hyperparameters can be tuned in the individual class files for each stage.
5. The datasets are imported from the Keras library. Maintain 'mnist' or 'fashion_mnist'
   everywhere as parameters to any functions in the code.

Note: To obtain the results enlisted in the paper, load the pretrained model and change the 
component number in GEC.py. Also, try changing the Nc hyperparameter in GEC.py to test the 
heirarchy of clusters.


##############################################################################################