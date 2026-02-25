import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from config import ConfigDataset, ConfigPlot
from dataset.math.plot import DatasetPlot
from dataset.gen.dataframe import DataframeGenerator

class DatasetGenerator:
    def generate(self, paths, labels, indicies, dataset_config, plot_config, image_processing):
        """
        Generates train, validation and test data iterators for each picture (tensorflow iterator). 
    
        Args:
            paths (str): Paths to all directories, which includes images.
            labels (str): List of labels to corresponding path to directory.
            indicies list([int,int], [int,int]) : List of range, data outside range is removed. Blank removes nothing from original data.
            dataset_config (ConfigDataset): Plot config, target size of reformatting of images in datasets.
            plot_config (ConfigPlot): Dataset config, target size of reformatting of images in datasets.
            image_processing (func): Image processing function when loading image. 

        Returns:
            _type_: train, validation and test data. Iterator (tensorflow) for images.
        """
        
        df = DataframeGenerator.generate(paths, labels, indicies)
        
        if(dataset_config.debug):
            print("\nDataframe:")
            print(df.head())
            print("")
            print(df.tail())
            print("")
            
        train_df, test_df = train_test_split(df, 
                                             test_size=dataset_config.split, 
                                             random_state=42,
                                             stratify=df.labels)
        train_df, val_df = train_test_split(train_df, 
                                            test_size=(dataset_config.split-0.1), 
                                            random_state=42,
                                            stratify=train_df.labels)
        
        if(dataset_config.debug):
            print("\nDataframe shapes (split): \n")
            print("Training dataset   : ", train_df.shape)
            print("Test dataset       : ",test_df.shape)
            print("Validation dataset : ", val_df.shape)
            print("")
        
        # Fix pre-proccessing for single dataset. 
        if(image_processing != None):
            image_gen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=lambda image: self.default_processing(image)
        )
        else:      
            image_gen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=lambda image: self.default_processing(image)
        )
        
       # CHANGED COLOR MODE TO GRAYSCALE. 
        train_data = image_gen.flow_from_dataframe(dataframe=train_df,x_col="filepaths",y_col="labels",
                                            target_size=(dataset_config.image_size,dataset_config.image_size),
                                            color_mode='grayscale',
                                            class_mode="categorical", 
                                            batch_size=dataset_config.batch_size,
                                            shuffle=True)
        
        validation_data = image_gen.flow_from_dataframe(dataframe=val_df,x_col="filepaths", y_col="labels",
                                            target_size=(dataset_config.image_size,dataset_config.image_size),
                                            color_mode= 'grayscale',
                                            class_mode="categorical",
                                            batch_size=dataset_config.batch_size,
                                            shuffle=False)
        
        test_data = image_gen.flow_from_dataframe(dataframe=test_df,x_col="filepaths", y_col="labels",
                                            target_size=(dataset_config.image_size,dataset_config.image_size),
                                            color_mode='grayscale',
                                            class_mode="categorical",
                                            batch_size=dataset_config.batch_size,
                                            shuffle=False)
        
        if(plot_config.plot and dataset_config.debug):
            plot = DatasetPlot()
            plot.plot_classes(train_data, train_df, plot_config)
        
        return train_data, validation_data, test_data

    def default_processing(self, image):
        """
        Default image processing function when none provided in generate.

        Args:
            image (_type_): Image that gets processed.

        Returns:
            _type_: _description_
        """
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]

        image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)

        image = np.clip(image * 1.25, 0, 255)

        # Restore channel dimension so shape is (H, W, 1)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        return image