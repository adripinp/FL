import tensorflow as tf
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import ConfigFederated, ConfigOod
from federated.client.clients import Clients
from ood.hdff import Hdff

class FederatedPlot:
    def plot_ood(self, hdff: Hdff, federated_config : ConfigFederated, ood_config : ConfigOod, xlabel : str, title : str):
        plt.figure(num=title, figsize=(14, 5))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1)) 
        
        plt.plot(hdff.results, label="Local model " + "(" + str(id) + ")")

        plt.xlabel(xlabel)
        plt.ylabel('Cosine (arg), Model vs. Global Model')
        plt.title(title + " (" + "Security mechanism="+str(ood_config.ood_protection) + ")")
        plt.plot([min(plt.xlim()),max(plt.xlim())],[float(ood_config.ood_protection_thres),float(ood_config.ood_protection_thres)], 'k--', label="OOD threshold")
        plt.legend()
        
    def plot_ood_dict(self, result : dict, federated_config : ConfigFederated, ood_config : ConfigOod, xlabel : str, title : str):
        plt.figure(num=title, figsize=(14, 5))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1)) 
        for id in range(0, federated_config.clients): 
            if(id == federated_config.host_id):
                plt.plot(result[id], label="Global model " + "(" + str(id) + ")")
            else:
                plt.plot(result[id], label="Local model " + "(" + str(id) + ")")

        plt.xlabel(xlabel)
        plt.ylabel('Cosine (arg), Model vs. Global Model')
        plt.title(title + " (" + "Security mechanism="+str(ood_config.ood_protection) + ")")
        plt.plot([min(plt.xlim()),max(plt.xlim())],[float(ood_config.ood_protection_thres),float(ood_config.ood_protection_thres)], 'k--', label="OOD threshold")
        plt.legend()