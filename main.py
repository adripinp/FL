import os
import tensorflow as tf
import random
import numpy as np

from dataset.dataset import Dataset
from config import ConfigFederated, ConfigOod, ConfigModel, ConfigDataset, ConfigPlot
from dataset.download.a_faces16000 import Afaces16000
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_alzheimer5100_poisoned import Balzheimer5100_poisoned
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
from dataset.download.l_pneumonia5200 import Lpneumonia5200
from federated.federated import Federated
from model.model import Model

from model.math.plot import ModelPlot

#
# FEEL FREE TO EDIT THE CONTENT OF ALL GIVEN FILES AS YOU LIKE.
#

############# REPRODUCIBILITY, deterministic behavior #############
def set_seeds(SEED):
    """ Set seeds for deterministic behavior. 
    """
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

def set_global_determinism(SEED):
    set_seeds(SEED=SEED)
    
    # Uncomment below if limiting cpu treads, may help with determinism. However may slow down training, especially on large models.
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # CUDA/GPU users
    # os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # tf.config.experimental.enable_op_determinism()

SEED = 42   # Change seed as you like. 
set_global_determinism(SEED=SEED)
###############################################################

class ModelSimulation():
    """ Only for testing single model. 
    """
           
    def run(self):
        #-----------CONFIG--------------------
        model_config = ConfigModel(
            debug = True,
            epochs = 5,
            activation = 'relu',
            activation_out = 'softmax',
            optimizer = 'adam',
            loss = 'categorical_crossentropy'
        )
        dataset_config = ConfigDataset(
            debug = True,                   # DISABLE IF YOU WANT TO PREVENT IMAGE EXAMPLES FROM BEING DISPLAYED BEFORE TRAINING.
            batch_size = 64,                          
            image_size = 256,            
            input_shape = (256,256,1),   
            split = 0.25,
            number_of_classes=2
        )
        plot_config = ConfigPlot(
            plot = True,
            path = "./.env/.saved/",
            img_per_class = 10  
        )
        
        #-----------SIM---------------------- 
        m = Model(
            model_config=model_config,
            dataset_config=dataset_config,
            plot_config=plot_config
        )
        
        # Create dataset by merging multiple datasets together. 
        train_data, validation_data, test_data = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),
                (Btumor3000().ID, Btumor3000(), []),
                (Balzheimer5100().ID, Balzheimer5100(), []),
                (Lpneumonia5200().ID, Lpneumonia5200(), [])
                
            ],
            dataset_config=dataset_config,
            plot_config=plot_config
        ).mergeAll()
        
        # Below is an example of dataset with subsets. That can be used in federated learning context. 
        # All given datasets down below are available in ./dataset/download.
        #
        # dataset = Dataset(
        #     [
        #         (Btumor4600().ID, Btumor4600(), []),    # id 0 ID DATA 
        #         (Btumor3000().ID, Btumor3000(), []),    # id 1 ID DATA 
        #         (Balzheimer5100().ID, Balzheimer5100(), []), # id 2 ID DATA 
        #         (Lpneumonia5200().ID, Lpneumonia5200(), []), # id 3 ID DATA 
        #         (Lpneumonia5200().ID, Lpneumonia5200(), [[300,500],[3600,3800]]), # id 4 ID DATA (subsamples of total, not used in pre-training) 
        #         (Btumor4600().ID, Btumor4600(), [[300,500],[3700,3900]]), # id 5 ID DATA (subsamples of total, not used in pre-training)  
        #         (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), [[1000,1700],[4000,4700]]), # id 6 OOD DATA (poisoned data) (not used in pre-training)
        #         (Afaces16000().ID, Afaces16000(), [[1,700],[2501,3200]]) # id 7 OOD DATA (not used in pre-training) # Take some two subsets of complete dataset. # [250,750], [4000,4500]
        #     ],
        #     dataset_config=dataset_config,
        #     plot_config=plot_config
        # )
        #
        # Bind local model / client to dataset id / subsets.
        #
        # train_data, validation_data, test_data = dataset.get(index i)
        #
        
        m.train(train_data, validation_data)     
        m.test(test_data) 
        
        m.plot_all(test_data, xlabel="CNN Model", title="CNN Model")
        input('Press Enter to close plots and exit...')
        
class FederatedSimulation():
    """
        TODO:
        Simulation for federated learning with multiple local models.
        You are free to change the configuration as you like, remove or edit.
        
        This are only some parameters for simulation that might be of interest.
        
        You may remove the configuration entirely if you prefer and apply an alternative setup.
    """
    #--------------------------------------- 
    #--------------CONFIG-------------------
    #--------------------------------------- 
    federated_config = ConfigFederated(
        debug = True,
            
        # _________FILE____________
        save = False,                         # IF save global model after sim. 
        load_round = 0,                       # Which specific global model from directory (directory id).
        load_reg = True,                      # Regression from global model after loading pretrained or not.
        load = False,                         # IF load from directory.
        delete_on_load = False,               # Delete loaded model in directory.
        path = "./.env/.saved/",              # Path to saved global model.
            
        # _______SIMULATION________
        rounds = 25,                          # Number of rounds for sim in federated. 
        ood_round = 26,                       # Round where ood starts.  
        clients = 5,                          # Number of clients in sim. (global model + local models).
        participants = 4,                     # How many participants for current round (randomized between clients). (local models only)
        host_id=0,                            # Host id (index). Should probably always be 0 (global model).
        client_to_dataset=[[0,1,2,3],[0],[1],[2],[3]]   # local model -> dataset assignment (array index in dataset).
    )
    
    ood_config = ConfigOod(
        debug = True,
        hdc_debug = False,
            
        # _______SIMULATION________
        enabled = False,                            # Enabling hdff and ood. 
        hyper_size=int(1e4),                        # Hyper dimensions for projection matrix. 
        
        id_client = [1,2,3,4],                      # Clients id in-distribution, excluding global client/model.
        ood_client = [5,6],                           # Datasets out-of-distribution (index in dataset), excluding global client/model.
        
        ood_protection = True,                     # IF ood protection (exluding) is enabled.
        ood_protection_thres = 0.7                  # Threshold for consider models being ood. 
    )
    
    model_config = ConfigModel(
        debug = True,
            
        # _______SIMULATION________
        epochs = 1,                         # Highly recommend to be 1 for federated. Change rounds instead. 
        activation = 'relu',                # Parameters for model.
        activation_out = 'softmax',
        optimizer = 'adam',
        loss = 'categorical_crossentropy'   # 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug = False,
            
        # _______SIMULATION________
        batch_size = 64,                    # Batchsize for datasets.                          
        image_size = 256,                   # Image size 
        input_shape = (256,256,1),
        split = 0.25,                       # Train / validation split.
        number_of_classes = 2               # Number of total classes.
    )
    plot_config = ConfigPlot(
        plot = False,
            
        # __________FILE___________
        path = './.env/plot',                
        img_per_class = 10                  # Only for plotting example pictures from dataset (debug must be set to true).
    )
    
    def run(self):
        #------------------------------------ 
        #--------------SIM------------------- 
        #------------------------------------ 
        m = Model(
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        # client-to-dataset and id dataset and ood dataset are referenced to this list. 
        dataset = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),    # id 0 ID DATA 
                (Btumor3000().ID, Btumor3000(), []),    # id 1 ID DATA 
                (Balzheimer5100().ID, Balzheimer5100(), []), # id 2 ID DATA 
                (Lpneumonia5200().ID, Lpneumonia5200(), []), # id 3 ID DATA 
                (Lpneumonia5200().ID, Lpneumonia5200(), [[300,500],[3600,3800]]), # id 4 ID DATA (subsamples of total, not used in pre-training) 
                (Btumor4600().ID, Btumor4600(), [[300,500],[3700,3900]]), # id 5 ID DATA (subsamples of total, not used in pre-training) 
                (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), [[1000,1700],[4000,4700]]), # id 6 OOD DATA (poisoned data) (not used in pre-training)
                (Afaces16000().ID, Afaces16000(), [[1,700],[2501,3200]]) # id 7 OOD DATA (not used in pre-training) # Take some two subsets of complete dataset. # [250,750], [4000,4500]
            ],
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        federated = Federated(
            dataset=dataset, 
            model=m,
            federated_config=self.federated_config,
            ood_config=self.ood_config, 
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        return federated.run()
    
if __name__ == "__main__": 
    # Model. 
    
    #-------------------------
    sim1 = ModelSimulation()
    sim1.run()