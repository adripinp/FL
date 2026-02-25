import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

from config import ConfigOod
from dataset.dataset import Dataset
from model.model import Model

class OodScore:
    """ Calculates AUROC and AUPR scores based on model, and ood / id data. """
    
    def __init__(self, ood_config: ConfigOod) -> None:
        self.ood_config = ood_config
        self.avg_auroc_model = []
        self.avg_aupr_model = []
        self.avg_auroc_similarity = []
        self.avg_aupr_similarity = []
    
    def get_softmax_scores(self, model, generator):
        """Computes max softmax confidence scores for a dataset."""
        scores = []
        ground_truth = []
        for images, labels in generator:  # Ignore labels, use images only
            logits = model.predict(images, verbose=0)
            softmax_scores = tf.nn.softmax(logits, axis=1)  # Compute softmax
            max_scores = tf.argmax(softmax_scores, axis=1).numpy()  # Take max prob
            scores.extend(max_scores)
            
            numeric_labels = tf.argmax(labels, axis=1).numpy()
            ground_truth.extend(numeric_labels)
            
            # Stop when all images are processed
            if len(scores) >= generator.n:
                break
        
        return scores, ground_truth
    
    def compute_auroc_model(self, model, generator, ood_config: ConfigOod):
        model_scores, ground_truth = self.get_softmax_scores(model, generator)
        
        # Debug prints
        #print(f"Model Scores: {model_scores}")
        #print(f"Ground Truth: {ground_truth}")
        
        auroc = roc_auc_score(ground_truth, model_scores)
        self.avg_auroc_model.append(auroc)

        print()
        print("Model scores AUROC")
        print(f"\nAUROC SCORE = {auroc:.4f}")
        
        avg_auroc = np.mean(self.avg_auroc_model)
        print(f"AUROC SCORE (AVG) = {avg_auroc:.4f}\n")
        
        return avg_auroc
    
    def compute_aupr_model(self, model, generator, ood_config: ConfigOod):
        model_scores, ground_truth = self.get_softmax_scores(model, generator)
        
        # Debug prints
        #print(f"Model Scores: {model_scores}")
        #print(f"Ground Truth: {ground_truth}")

        aupr = average_precision_score(ground_truth, model_scores)
        self.avg_aupr_model.append(aupr)
        
        print("Model scores AUPR")
        print(f"\nAUPR SCORE = {aupr:.4f}")
        
        avg_aupr = np.mean(self.avg_aupr_model)
        print(f"AUPR SCORE (AVG) = {avg_aupr:.4f}\n")
        
        return avg_aupr
    
    def compute_auroc_similarity(self, similarity, ood_config: ConfigOod):
        """_summary_

        Args:
            similarity (list): list with similairty scores of model, from HDFF.
            ood_config (ConfigOod): _description_

        Returns:
            list: AUROC score, averaging siimilarity scores. 
        """
        similarity_scores = np.array(similarity)
        
        ground_truth = [1] * len(ood_config.id_client) + [0] * len(ood_config.ood_client)
        ground_truth = np.array(ground_truth)
        
        # Debug prints
        # print(f"Similarity Scores: {similarity_scores}")
        # print(f"Ground Truth: {ground_truth}")
        
        auroc = roc_auc_score(ground_truth, similarity_scores)
        self.avg_auroc_similarity.append(auroc)
        
        print("Similarity scores AUROC")
        print(f"\nAUROC SCORE = {auroc:.4f}")
        
        avg_auroc = np.mean(self.avg_auroc_similarity)
        print(f"AUROC SCORE (AVG) = {avg_auroc:.4f}\n")
        
        return avg_auroc
        
    def compute_aupr_similarity(self, similarity, ood_config: ConfigOod):
        similarity_scores = np.array(similarity)
        ground_truth = [1] * len(ood_config.id_client) + [0] * len(ood_config.ood_client)
        ground_truth = np.array(ground_truth)

        # Debug prints
        # print(f"Similarity Scores: {similarity_scores}")
        # print(f"Ground Truth: {ground_truth}")

        aupr = average_precision_score(ground_truth, similarity_scores)
        self.avg_aupr_similarity.append(aupr)

        print("Similarity scores AUPR")
        print(f"\nAUPR SCORE = {aupr:.4f}")
        
        avg_aupr = np.mean(self.avg_aupr_similarity)
        print(f"AUPR SCORE (AVG) = {avg_aupr:.4f}\n")
        
        return avg_aupr