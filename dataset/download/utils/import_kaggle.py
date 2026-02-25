import cv2
import kagglehub
import numpy as np

class ImportKaggle():
    paths = None
    labels = None
    
    def __init__(self, ID, KAGGLE_PATH, STATIC_PATH, paths, labels) -> None:
        """
            Import utility for importing datasets from kaggle.
        Args:
            ID (str): Identifier for dataset (can be any)
            KAGGLE_PATH (str): Path to dataset on kaggle site, (last in url).
            STATIC_PATH (str): Path to subfolder's with images on local machine, (folder strucutre may be different from datasets).
            paths (str): Name of subfolders (as path).
            labels (str): Target labels, what new labels should be for all images in subfolders, same length as paths, position in labels correspond to postion in paths. 
        """
        self.ID = ID    # Identifier in debug for name of dataset. 
        
        print("Downloading dataset..." + self.ID + '\n')
        
        PATH = kagglehub.dataset_download(KAGGLE_PATH)
        print("Path to dataset files:", PATH)
        print()
        
        PATH = PATH + STATIC_PATH  # Must specifiy path to where path of subfolders of pictures exist. 
        
        paths_result = []          # Construct seperate paths to each subfolder with images. 
        for x in paths:
            paths_result.append([PATH + x])
        
        self.paths = paths_result
        self.labels = labels
        
    def pre_processing(image):
        """
            Basic pre processing function. Can be overrided in inheritance. Currying in dataset generator. 
        Args:
            image (img): Image as input when currying.

        Returns:
            img: processed image. 
        """
        image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)

        value = image[:, :, 2]
        value = np.clip(value * 1.25, 0, 255)

        image[:, :, 2] = value

        return image