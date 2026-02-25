from dataset.download.utils.import_kaggle import ImportKaggle

class Btumor4600(ImportKaggle):  
    KAGGLE_PATH = ""
    STATIC_PATH = ""
    ID = ""
    
    def __init__(self) -> None:  
        """
            Structure for downloading dataset. Inheritance from utils. Kaggle dataset -> ImportKaggle.py . 
        """
        
        # https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset
        KAGGLE_PATH = "preetviradiya/brian-tumor-dataset"
        
        STATIC_PATH = '/Brain Tumor Data Set/Brain Tumor Data Set'  # path in folders to folders with images. 
        paths = ['/Brain Tumor', '/Healthy']
        labels = ['Sick', 'Healthy']             
           
        ID = __name__              
        
        super().__init__(ID=ID, KAGGLE_PATH=KAGGLE_PATH, STATIC_PATH=STATIC_PATH, paths=paths, labels=labels)
    




