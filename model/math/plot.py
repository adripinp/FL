import tensorflow as tf
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class ModelPlot:
    def accuracy(self, train_accuracy, val_accuracy, train_loss, val_loss, xlabel : str,  title : str):
        # Create a figure with two subplots: one for accuracy and one for loss
        plt.figure(num=title, figsize=(14, 5))

        # Plot training and validation accuracy
        plt.subplot(2, 3, 4)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   # Force integers on x-axis
        plt.plot(train_accuracy, label='Training Accuracy', color='blue')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   # Force integers on x-axis
        plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
        plt.xlabel(xlabel)
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy ' + title)
        plt.legend()

        # Plot training and validation loss
        plt.subplot(2, 3, 5)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   # Force integers on x-axis
        plt.plot(train_loss, label='Training Loss', color='blue')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   # Force integers on x-axis
        plt.plot(val_loss, label='Validation Loss', color='orange')
        plt.xlabel(xlabel)
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss ' + title)
        plt.legend()

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.ion()  # to make plot non-blocking, i.e. if multiple plots are launched

        # Display the plot
        plt.show()
        
    def test_accuracy(self, accuracy, loss, xlabel : str, title : str):
        plt.figure(num=title, figsize=(14, 5))

        # Plot test
        plt.subplot(2, 3, 1)
        
        test_rounds = []
        for i in range(len(accuracy)):
            test_rounds.append(i)
        
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   # Force integers on x-axis
        plt.plot(test_rounds, accuracy, label='Test Accuracy', color='blue')
        plt.xlabel(xlabel)
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy ' + title)
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   # Force integers on x-axis
        plt.plot(test_rounds, loss, label='Test Loss', color='blue')
        
        plt.xlabel(xlabel)
        plt.ylabel('Loss')
        plt.title('Test Loss ' + title)
        plt.legend()

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.ion()  # to make plot non-blocking, i.e. if multiple plots are launched

        # Display the plot
        plt.show()
        
        
    def confusion_matrix(self, model, test_data, title : str):
        # Get predictions for the test data
        y_pred = model.predict(test_data)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Get true labels for the test data
        true_classes = test_data.classes

        from sklearn.metrics import confusion_matrix
        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_classes, y_pred_classes)

        plt.figure(num=title, figsize=(14, 14))
        plt.subplot(2, 3, 3)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', square=True, 
                    xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
        plt.xlabel('Predicted label')
        plt.ylabel('Actual label')
        plt.title('Confusion Matrix ' + title)
        plt.ion()  # to make plot non-blocking, i.e. if multiple plots are launched
        
        plt.show()