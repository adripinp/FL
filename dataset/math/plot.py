import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DatasetPlot:
    def plot_classes(self, dataset, dataset_df, plot_config):
        # Get class labels
        class_labels = list(dataset.class_indices.keys())

        # Plot images from each class
        plt.figure(figsize=(15, 10))
        for label in class_labels:
            # Get indices of images belonging to the current class
            indices = dataset_df[dataset_df['labels'] == label].index
            
            # Randomly sample a subset of indices if there are more than the desired number of images per class
            indices = np.random.choice(indices, min(plot_config.img_per_class, len(indices)), replace=False)
            
            # Plot images
            for i, idx in enumerate(indices):
                plt.subplot(len(class_labels), plot_config.img_per_class, len(class_labels)*i + class_labels.index(label) + 1)
                plt.imshow(plt.imread(dataset_df.loc[idx, 'filepaths']))  
                plt.title(label)
                plt.axis('off')
        plt.show()

    