from dataset.download.utils.import_kaggle import ImportKaggle

class Lpneumonia5200(ImportKaggle):
    PATH = ""
    STATIC_PATH = ""
    ID = ""
    
    def __init__(self) -> None:
        """
            Structure for downloading dataset. Inheritance from utils. Kaggle dataset -> ImportKaggle.py . 
        """
        
        # https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
        KAGGLE_PATH = "paultimothymooney/chest-xray-pneumonia"
        
        STATIC_PATH = '/chest_xray/train'  # path in folders to folders with images. 
        paths = ['/NORMAL', '/PNEUMONIA']
        labels = ['Healthy', 'Sick']            
           
        ID = __name__              
        
        super().__init__(ID=ID, KAGGLE_PATH=KAGGLE_PATH, STATIC_PATH=STATIC_PATH, paths=paths, labels=labels)