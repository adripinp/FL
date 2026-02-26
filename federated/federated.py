import os

import tensorflow as tf

import random
import copy

from config import ConfigDataset, ConfigFederated, ConfigOod, ConfigPlot
from dataset.dataset import Dataset
from federated.math import federated_math

from ood.hdff import Hdff
from model.model import Model

class Federated():
    """
        Federated learning environment. Three cycles per round, update local models from global model, train local models, regression on local models and update global. 
    """
    def __init__(self, dataset : Dataset, model : Model, federated_config : ConfigFederated, ood_config : ConfigOod, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Args:
            dataset (Dataset): dataset, custom.
            model (Model): nn. model.
            federated_config (ConfigFederated): configuration for federated learning env. 
            ood_config (ConfigFederated): configuration for ood detection.
            dataset_config (ConfigDataset) : configuration for dataset. 
            plot_config (ConfigPlot): configuration for plotting.
        """
        self.dataset = dataset
        self.init_model = model            # This model can be 
        self.federated_config = federated_config
        self.ood_config = ood_config
        self.plot_config = plot_config
        
        self.models: dict[int, Model] = {}
        self.data: dict[int, tuple] = {}  # id -> (train, val, test)

        self._init_models(dataset_config=dataset_config, plot_config=plot_config)
        self._init_datasets()
    
    def run(self): 
        """
            Runs federated learning environment.
        """
        round_id = 0
        # load pre-trained snapshot
        if self.federated_config.load:
            round_id = int(self.federated_config.load_round)
            self._load_models(round_id=round_id)
        
        if round_id < int(self.federated_config.rounds):  
            last_round = self.train_(round_id)
            self.result(title_suffix=f"_round{last_round}")  # show Phase 1 plots (global test curve + confusion matrix)
            # final save
            if self.federated_config.save:
                self._save_models(round_id=last_round)
        else:                                            # Only test if round = round on specific clients in load. 
            self.test_()
            
        return None
  
    def train_(self, start : int):
        host = int(self.federated_config.host_id)
        clients = int(self.federated_config.clients)
        total_rounds = int(self.federated_config.rounds)
        
        for round_id in range(1+start, int(self.federated_config.rounds) + 1):   

            # All is eligibale except host
            eligible = [i for i in range(clients) if i != host]

            # (Phase 1 usually has ood_config.enabled=False, but keep logic correct)
            if self.ood_config.enabled and round_id < int(self.federated_config.ood_round):
                eligible = [i for i in eligible if i not in self.ood_config.ood_client]

            part = max(int(self.federated_config.participants), 1)                 # Alteast one client will partcipate in round.
            part = min(part, len(eligible))  # prevent sample crash
            selected_clients = random.sample(eligible, part) 
            

            
            print(f"\n=== Round {round_id}/{total_rounds} | participants={selected_clients} ===")

            # 1) Regression: global -> locals
            self.global_(host, round_id)                     # Update all local models with global model.

            # 2) Local training
            for id in selected_clients:                                            # Train all local models. 
                self.local_(id, round_id)

            # 3) Aggregation + 4) Global eval
            self.update_(selected_clients, round_id)
    
        return total_rounds
    
    def test_(self):
        host = int(self.federated_config.host_id)
        _, _, test_data = self.data[host]
        print("Evaluating global model...")
        self.models[host].test(test_data)
        self.result()
    
    def global_(self, host_id: int, round_id: int):                                            # Update all local models with global model weights. 
        """
            Updates local models with global model weights. 
        Args:
            id (int): id for global model.
        """
        # skip regression immediately after loading
        if (
            self.federated_config.load
            and (not self.federated_config.load_reg)
            and round_id == int(self.federated_config.load_round) + 1
        ):
            return
        
        global_model = self.models[host_id].model
        assert global_model is not None
        global_weights = global_model.get_weights()

        for mid, m in self.models.items():
            if mid == host_id:
                continue
            local_model = m.model
            assert local_model is not None
            local_model.set_weights(global_weights)
            
    def local_(self,  client_id: int, round_id : int):                                         # Train local models
        """
            Trains local models, with id. 
        Args:
            id (int): id for local model. 
            round (int): current round. 
        """
        train_data, val_data, _ = self.data[client_id]

        if self.federated_config.debug:
            print(f"[Round {round_id}] local train: client {client_id}")

        self.models[client_id].train(train_data, val_data)

        
    def update_(self, selected_clients: list[int], round_id: int):
        """
            Update global model with clients that was training during round (selected clients).
            
            Incorporate ood detection if enabled in config. Select only clients that are not detected as ood.
        Args:
            selected_clients (list): list with id of clients (local models) that selected for training.
            round (int): current round. 
        """
        host = int(self.federated_config.host_id)

        # Phase 1: no OOD filtering
        participating = list(selected_clients)
        if not participating:
            print(f"[Round {round_id}] No clients to aggregate , skipping.")
            return

        # FedAvg
        client_weights = []
        for cid in participating:
            cm = self.models[cid].model
            assert cm is not None
            client_weights.append(cm.get_weights())
        
        new_global = federated_math.federated_mean(client_weights)
        
        gm = self.models[host].model
        assert gm is not None
        gm.set_weights(new_global)

        # Global eval on ALL ID test data (host is mapped to all ID datasets)
        _, _, test_data = self.data[host]
        if self.federated_config.debug:
            print(f"[Round {round_id}] global eval: server {host} on all ID test data")

        self.models[host].test(test_data)
        
        
    def ood_extraction(self, id : int, model : Model):
        """ Exctract features from model. 

        Args:
            id (int): Id of model.
            model (Model): Model (object).
        """
        # TODO
        
        return None
        
    def ood_detection(self, selected_clients):
        """ Detecting model being ood from selected clients that trains.

        Args:
            selected_clients (int): clients that undergo training this round.
        """
        # TODO
        
        return None
            
    def result(self, title_suffix=""):
        """
            Plot performance of each model. 
        """
        host = int(self.federated_config.host_id)
        _, _, test_data = self.data[host]

        if self.plot_config.plot:
            title = f"Server {host}{title_suffix}"
            self.models[host].plot_test(xlabel="Round", title=title)
            self.models[host].plot_all(test_data, xlabel="Round", title=title)
    

    # ---------------- init / save / load ----------------

    def _init_models(self, dataset_config: ConfigDataset, plot_config: ConfigPlot):
        for mid in range(int(self.federated_config.clients)):
            try:
                m = copy.deepcopy(self.init_model)
            except Exception:
                m = Model(self.init_model.model_config, dataset_config, plot_config)

            # don’t spam summaries for every client
            m.model_config.debug = False
            self.models[mid] = m

    def _init_datasets(self):
        assert self.federated_config.client_to_dataset is not None, "client_to_dataset must be set in ConfigFederated"
        for mid in range(int(self.federated_config.clients)):
            ds_ids = self.federated_config.client_to_dataset[mid]
            if ds_ids is None:
                raise ValueError(f"client_to_dataset[{mid}] is None")
            train_data, val_data, test_data = self.dataset.get(ds_ids)
            self.data[mid] = (train_data, val_data, test_data)

    def _model_path(self, model_id: int, round_id: int) -> str:
        os.makedirs(self.federated_config.path, exist_ok=True)
        return os.path.join(self.federated_config.path, f"model{model_id}_round{round_id}.keras")

    def _save_models(self, round_id: int):
        for mid, m in self.models.items():
            path = self._model_path(model_id=mid, round_id=round_id)
            km = m.model
            assert km is not None
            km.save(path)
            if self.federated_config.debug:
                print(f"Saved model {mid} -> {path}")

    def _load_models(self, round_id: int):
        for mid in range(int(self.federated_config.clients)):
            path = self._model_path(model_id=mid, round_id=round_id)
            if os.path.exists(path):
                loaded = tf.keras.models.load_model(path)
                loaded.compile(
                    optimizer=self.models[mid].model_config.optimizer,
                    loss=self.models[mid].model_config.loss,
                    metrics=["accuracy"],
                )
                self.models[mid].model = loaded

                if self.federated_config.debug:
                    print(f"Loaded model {mid} <- {path}")

                if self.federated_config.delete_on_load:
                    os.remove(path)