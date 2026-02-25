from dataset.download.utils.import_kaggle import ImportKaggle

class Btumor3000(ImportKaggle):
    KAGGLE_PATH = ""
    STATIC_PATH = ""
    ID = ""
    
    def __init__(self) -> None:
        """
            Structure for downloading dataset. Inheritance from utils. Kaggle dataset -> ImportKaggle.py . 
        """
        
        # https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri
        KAGGLE_PATH = "abhranta/brain-tumor-detection-mri"
        
        STATIC_PATH = '/Brain_Tumor_Detection'  # path in folders to folders with images. 
        paths = ['/yes', '/no']
        labels = ['Sick', 'Healthy']             
           
        ID = __name__              
        
        super().__init__(ID=ID, KAGGLE_PATH=KAGGLE_PATH, STATIC_PATH=STATIC_PATH, paths=paths, labels=labels)
      
        