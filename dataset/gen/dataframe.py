import numpy as np
import pandas as pd
import os 

class DataframeGenerator:
    def generate(paths, labels, indicies) -> pd.DataFrame:
        """
        Generates dataframe, holds paths to corresponding image and label in corresponding order.
        
        Parameters
        ----------
        paths : list
            List of paths to subdirectories (strings).
        labels : list
            List of labels in corresponding order of paths. What specific img in subdirectory will be labeled. 
        indicies : list
            List of ranges, data outside range is cut off. Empty list removes no data. 
        
        Returns
        ----------
        pd.DataFrame
            Dataframe holding corresponding paths and labels to directory. (Total)
        """
        
        result_paths = []
        result_labels = []
        
        for i, dir_list in enumerate(paths):
            for j in dir_list:
                    list_f = os.listdir(j)
                    for name in list_f:
                        fpath = os.path.join(j, name)
                        result_paths.append(fpath)
                        result_labels.append(labels[i])

        temp_paths = []
        temp_labels = []
        for x in indicies:
            if(0 <= x[0] < x[1] <= len(result_paths) and x != []):
                for y in result_paths[x[0]:x[1]]:
                    temp_paths.append(y)
                for y in result_labels[x[0]:x[1]]:
                    temp_labels.append(y)
            else:
                raise Exception("Indicies for dataset is incorrect.")
        if(temp_paths != [] and temp_labels != []):
            result_paths = temp_paths
            result_labels = temp_labels
        
        result_paths = pd.Series(result_paths, name="filepaths")
        result_labels = pd.Series(result_labels, name="labels")
        
        result_data = pd.concat([result_paths, result_labels], axis=1)
        
        return pd.DataFrame(result_data)