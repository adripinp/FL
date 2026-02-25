from dataset.download.utils.import_kaggle import ImportKaggle

class Afaces16000(ImportKaggle):
    PATH = ""
    STATIC_PATH = ""
    ID = ""
    
    def __init__(self) -> None:
        """
            Structure for downloading dataset. Inheritance from utils. Kaggle dataset -> ImportKaggle.py . 
        """
        
        # https://www.kaggle.com/datasets/andrewmvd/animal-faces
        KAGGLE_PATH = "andrewmvd/animal-faces"
        
        STATIC_PATH = '/afhq/train'  # path in folders to folders with images. 
        paths = ['/cat', '/dog']
        labels = ['Healthy', 'Sick']            
           
        ID = __name__              
        
        super().__init__(ID=ID, KAGGLE_PATH=KAGGLE_PATH, STATIC_PATH=STATIC_PATH, paths=paths, labels=labels)