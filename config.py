############################################################################################
#RECOMMENDED IS TO CHANGE CONFIGURATION IN SIMULATION INSTEAD OF DEFUALT VALUES GIVEN HERE.#
############################################################################################

import tensorflow as tf


exception_msg = "Config.py incorrect parameterized: "

class ConfigFederated():
    debug = False
    repeats = 1
    
    save = False
    load_round = 40                        # Which round should federated sim continue from (from disk).
    load = False
    delete_on_load = False
    path = "./.saved/"

    clients = 4                            # Clients (including server) in simulation.
    rounds = 5                             # Rounds of federated training.
    ood_round = 3
    participants = 3                       # participants = clients - 1; for all clients training each round without global model()
    host_id = 0
    client_to_dataset = [[0, 1, 2], [0], [1], [2]]   # Length should be same as clients. Clients load from corresponding position.
    
    def __init__(self, debug : bool, save : bool, load_round : int, load_reg : bool, load : bool, delete_on_load : bool, path : str, 
                 rounds : int, ood_round : int, clients : int, participants : int, host_id : int, client_to_dataset : list) -> None:
        """_summary_
        Federated learning configuration for simulation.

        Args:
            debug (bool): printout's in federated (more detailed).
            repeats (int): repeat simulation.
            save (bool): load and saves weights of trained clients after completed training.
            load_round (int): which round of saved model should be loaded.
            load (bool): if any model should be loaded from local disc. 
            delete_on_load (bool): delete loaded model from disk. 
            path (str): path to folder for saving client weights.
            rounds (int): rounds of training, iterations of client training.
            ood_round (int): round to start out of distribution detection. 
            clients (int): number of clients that will be created during sim.
            participants (int): number of participants that will train during round (select n random from clients). Have
            host_id (int): host id. 
            client_to_dataset (list[list[int]]): dataset allocation for clients. 

        """
        self.debug = debug
        
        self.save = save
        self.load = load
        self.load_reg = load_reg
        self.delete_on_load = delete_on_load
        self.path = path
        
        if(0 <= load_round < rounds):
            self.load_round = load_round
        elif (load):
            raise Exception(exception_msg, "Must load round less than sim. rounds.")
        
        if(clients >= 2):
            self.clients = clients
        else: 
            raise Exception(exception_msg,"Must have atleast two clients, one global model + one local model")
            
        if(rounds >= 1):
            self.rounds = rounds
        else:
            raise Exception(exception_msg, "Must run for atleast one round")
        
        if(ood_round >= 1):
            self.ood_round = ood_round
        else:
            raise Exception(exception_msg, "ood rounds must be greater than 0")
        
        if(participants < clients):
            self.participants = participants
        else:
            raise Exception(exception_msg, "Number of participants must be less than clients")
        
        if(0 <= host_id <= (clients-1)):
            self.host_id = host_id
        else:
            raise Exception(exception_msg, "Host id not valid ")
        
        if(len(client_to_dataset) == clients): 
            self.client_to_dataset = client_to_dataset
        else:
            raise Exception(exception_msg, "Not all clients are assigned a dataset, length of ")

class ConfigOod():
    debug = True
    hdc_debug = True
    enabled = False
    hyper_size = int(1e4)
    cosine_sum = tf.math.reduce_sum,
    id_client = []                    # Datasets in-distribution
    ood_client = []  
    
    ood_protection = False
    ood_protection_thres = 0.2

    def __init__(self, debug : bool, hdc_debug : bool, enabled : bool, hyper_size : int, id_client : list, ood_client : list, 
                 ood_protection : bool, ood_protection_thres : float):
        """ Configuration for hyperdimensional features.

        Args:
            debug (bool): More printout and prompts.
            hdc_debug (bool): More printout and prompts in hyperdimensional module.
            enabled (bool): Enables ood module.
            hyper_size (int): Dimension of hyperdim matrixes. 
            ood_protection (bool): IF models outside threshold should be disregarded. 
            ood_protection_thres (float): Threshold value for ood, less than equals ood. 

        Raises:
            Exception: _description_
            Exception: _description_
        """
        self.debug = debug
        self.hdc_debug = hdc_debug
        
        self.enabled = enabled
        self.id_client = id_client
        self.ood_client = ood_client
        
        self.ood_protection = ood_protection
        
        if(0 < hyper_size):
            self.hyper_size = hyper_size
        else:
            raise Exception(exception_msg, "Hyper size incorrect in hdff, must be >0")
        
        if(0 < ood_protection_thres < 1):
            self.ood_protection_thres = ood_protection_thres
        else:
            raise Exception(exception_msg, "ood protection threshold is incorrect, must be 0<x<1")

    
class ConfigModel():
    debug = True
    
    epochs = 15
    activation = 'relu'
    activation_out = 'softmax'
    optimizer = 'adam'
    loss = 'categorical_crossentropy'   # categorical_crossentropy
    
    def __init__(self, debug, epochs, activation, activation_out, optimizer, loss) -> None:
        """
        Model configuration for simulation.

        Parameters
        ----------
            debug : bool
                Printouts in model during execution.
            epochs : int
                Runs during training.
            activation : str
                Aciviation function for hidden layers (tensorflow.keras). 
            activation_out : str
                Activation function for output layer (tensorflow.keras).
            optimizer : str
                Optimizer function (tensorflow.keras).
            loss : str 
                Loss function (tensorflow.keras).
        """
        self.debug = debug
        
        if(self.epochs >= 1):
            self.epochs = epochs
        else:
            raise Exception(exception_msg, "Must atleast run for 1 epoch")
        self.activation = activation
        self.activation_out = activation_out
        self.optimizer = optimizer
        self.loss = loss

class ConfigDataset():
    debug = True
    
    batch_size = 64                          
    image_size = 256            # Target size.  256
    input_shape = (256,256,3)   # Target input shape. (256,256,3)
    
    split = 0.25                # 0.25. Test / validation portion of dataset. 
    
    def __init__(self, debug, batch_size, image_size, input_shape, split, number_of_classes) -> None:
        """
        Dataset configuration for simulation.

        Parameters
        ----------
            debug : bool
                Printouts in dataset during execution.
            batch_size : int
                Batch size for training. 
            image_size : int
                Target size for image generation.
            input_shape : tuple(int,int,int)
                Target input shape for image generation and model
            split : (float) 
                How train, validation and test data will be splitted. 
        """
        
        self.debug = debug
        self.batch_size = batch_size
        
        if((image_size == input_shape[0]) and (image_size == input_shape[1])):
            self.image_size = image_size
            self.input_shape = input_shape
        else:
            raise Exception(exception_msg, "Dataset dimensions incorrect")
        
        if(0 < split < 1):
            self.split = split
        else:
            raise Exception(exception_msg, "split is percentage in float")
        
        if(number_of_classes > 0):
            self.number_of_classes = number_of_classes
        else:
            raise Exception(exception_msg, "number of classes must be greater than 0")
            
class ConfigPlot():
    plot = True
    path = './.env/plot'
    img_per_class = 10          # Images per class that will be plotted. 
    
    def __init__(self, plot : bool, path : str, img_per_class : int) -> None:
        """
        Plot configuration for simulation.
        
        Parameters
        ----------
            plot : bool
                If plotting enabled during execution.
            path : str
                Path to save certain plots. 
            img_per_class : int
                Image per class that gets showed during plot of dataset classes.
        """
        self.plot = plot
        self.path = path
        self.img_per_class = img_per_class