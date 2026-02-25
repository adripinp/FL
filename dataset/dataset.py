from config import ConfigDataset, ConfigPlot
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
from dataset.generator import DatasetGenerator

class Dataset:
    datasets = []
    dataset_config = None
    plot_config = None
    
    def __init__(self, datasets : list, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Packages all downloaded datasets.
        
        Parameters
        ----------
            datasets : list[(str(ID),dataset, [int,int])]
                List containing tuple of id (str), dataset and range of samples.
            dataset_config : ConfigDataset 
                Dataset config, target size of reformatting of images in datasets.
            plot_config : ConfigPlot 
                Plotting config, plotting some img / labels for respective / merged dataset.
        """
        self.datasets = datasets
        self.dataset_config = dataset_config
        self.plot_config = plot_config
    
    def split_indicies(self, result_paths, result_labels, indicies):
        if(0 <= indicies[0] < indicies[1]):
            return result_paths[indicies[0]:indicies[1]], result_labels[indicies[0]:indicies[1]]
        else:
            raise Exception("Range in dataset is incorrect")
    
    def mergeAll(self):
        """
        Returns tensorflow iterator to directory of all datasets (merged). 
        
        Raises
        ----------
        Exception
            No datasets loaded.

        Returns
        ----------
        _type_ 
            train, validation, test. Iterator (tensorflow) to directory.
        """
        
        if len(self.datasets) == 0:
            raise Exception("No datasets loaded in DatasetGenerator.")
        for x in self.datasets:
            if len(x) != 3:
                raise Exception("Must specify name (string), dataset (ImportKaggle), indicies of dataset ([int,int])")
        
        result_paths = []
        result_labels = []
        result_indicies = []
        
        print(self.datasets)
        for x in self.datasets:
            print(x)
            for y in x[1].paths:
                result_paths.append(y)
            for y in x[1].labels:
                result_labels.append(y)
            result_indicies = result_indicies + x[2]
                
        dataset = DatasetGenerator()
        
        return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, None)
        
    def get(self, *args):
        """
        Returns tensorflow iterator to directory of dataset index i. 
        
        Raises
        ----------
        Exception
            No datasets loaded.

        Returns
        ----------
        _type_ 
            train, validation, test. Iterator (tensorflow) to directory.
        """
        result_paths = []
        result_labels = []
        result_indicies = []
        
        if len(args) == 1 and isinstance(args[0], int):
            i = args[0]
            if(0 <= i <= (len(self.datasets) - 1)):
                # Split if defined. 
                result_paths = self.datasets[i][1].paths
                result_labels = self.datasets[i][1].labels
                result_indicies = result_indicies + self.datasets[i][2]
                
                dataset = DatasetGenerator()
                
                return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, lambda image: self.datasets[i][1].pre_processing(image))
            else:
                raise Exception("Index out of range.")
        
        if len(args) == 1 and isinstance(args[0], list):
            for i in args[0]:
                if(0 <= i <= (len(self.datasets) - 1)):
                    result_paths = result_paths + self.datasets[i][1].paths
                    result_labels = result_labels + self.datasets[i][1].labels
                    result_indicies = result_indicies + self.datasets[i][2]
                else:
                    raise Exception("Index out of range.")
                
            dataset = DatasetGenerator()
            return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, lambda image: self.datasets[i][1].pre_processing(image))
        
    def print(self):
        """
            Prints all loaded datasets in generator.
        """
        for x in (self.datasets):
            print(x)
  


        