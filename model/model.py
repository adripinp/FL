import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten

from config import ConfigDataset, ConfigModel, ConfigPlot
from model.math.plot import ModelPlot

class Model:
    def __init__(self, model_config : ConfigModel, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Single model structure for training / classification. 

        Parameter
        ----------
            model_config : ConfigModel
                Model config, properties for neural network.
            dataset_config : ConfigDataset
                Dataset config, how input shape in model should be.
            plot_config : ConfigPlot
                Plotting config, plotting history and confusion matrix.
        """
        self.plot = ModelPlot()
        self.model = None
        
        self.epochs = 0         # How many epochs was runned for model (most useful for federated env. to see how much local model contributed to global model).
        
        self.acc = None         # For training / validation plotting.
        self.val_acc = None
        self.loss = None
        self.val_loss = None

        self.test_accuracy = [] # For test plotting. 
        self.test_loss = []
        
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.plot_config = plot_config
        
        # ------------------------------- MODEL -------------------------------
        
        model = Sequential()
            
        model.add(InputLayer(input_shape=self.dataset_config.input_shape, batch_size=self.dataset_config.batch_size))
        model.add(Conv2D(16, 16, activation=self.model_config.activation)) # 16
        model.add(MaxPooling2D())
            
        model.add(Conv2D(16, 16, activation=self.model_config.activation)) # 32 
        model.add(MaxPooling2D())
        
        model.add(Conv2D(16, 16, activation=self.model_config.activation)) # 32 
        model.add(MaxPooling2D())
            
        model.add(Conv2D(16, 16, activation=self.model_config.activation)) # 16
        model.add(MaxPooling2D())
            
        model.add(Flatten())
        model.add(Dense(32, activation=self.model_config.activation))        # 256
        model.add(Dense(2, activation=self.model_config.activation_out))
        
        # ---------------------------------------------------------------------
        
        model.compile(optimizer=self.model_config.optimizer, loss=self.model_config.loss, metrics=['accuracy'])
        
        if(self.model_config.debug):
            model.summary()
        
        self.model = model

    def train(self, train_data, val_data):
        """
        Trains model and evaluate with test data. 

        Parameters
        ----------
            train_data : _type_
                train data for training.
            val_data : _type_
                validation data for training.
        """
        
        history = self.model.fit(train_data,
                    epochs=int(self.epochs + self.model_config.epochs),
                    validation_data=val_data,
                    initial_epoch=self.epochs)
        
        if(self.epochs == 0):
            self.acc = history.history['accuracy']
            self.val_acc = history.history['val_accuracy']

            self.loss = history.history['loss']
            self.val_loss = history.history['val_loss']
        else:
            self.acc += history.history['accuracy']
            self.val_acc += history.history['val_accuracy']

            self.loss += history.history['loss']
            self.val_loss += history.history['val_loss']
            
        self.epochs = int(self.epochs + self.model_config.epochs)
        
    def test(self, test_data):
        loss, accuracy = self.model.evaluate(test_data) # Evaluate model on test data.
        
        self.test_accuracy.append(accuracy)
        self.test_loss.append(loss)

        if(self.model_config.debug):
            print("Test loss= ", loss)
            print("Test accuracy=", accuracy)
            print()
            
    def plot_all(self, test_data, xlabel : str, title : str):
        if(self.acc != None):
            self.plot.accuracy(self.acc, self.val_acc, self.loss, self.val_loss, xlabel, title)
        self.plot.confusion_matrix(self.model, test_data, title)
        
    def plot_test(self, xlabel : str, title : str):
        self.plot.test_accuracy(self.test_accuracy, self.test_loss, xlabel, title)
    