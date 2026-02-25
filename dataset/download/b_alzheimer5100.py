from dataset.download.utils.import_kaggle import ImportKaggle

class Balzheimer5100(ImportKaggle):
    PATH = ""
    STATIC_PATH = ""
    ID = ""
    
    def __init__(self) -> None:
        """
            Structure for downloading dataset. Inheritance from utils. Kaggle dataset -> ImportKaggle.py . 
        """
        
        # https://www.kaggle.com/datasets/yasserhessein/dataset-alzheimer
        KAGGLE_PATH = "yasserhessein/dataset-alzheimer"
        
        STATIC_PATH = '/Alzheimer_s Dataset/train'  # path in folders to folders with images. 
        paths = ['/VeryMildDemented', '/MildDemented', '/ModerateDemented', '/NonDemented']
        labels = ['Sick', 'Sick', 'Sick', 'Healthy']            
           
        ID = __name__              
        
        super().__init__(ID=ID, KAGGLE_PATH=KAGGLE_PATH, STATIC_PATH=STATIC_PATH, paths=paths, labels=labels)
        