import os
import re
import time
import pandas as pd
from copy import deepcopy
from typing import Dict

import torch
import yaml
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from SAnD.core.model import NARX_Transformer
from SAnD.utils.functions import generar_pares_aleatorios
from dataset import load_NASA
from tqdm import tqdm, trange


# from transformer import LEARNING_RATE


class NeuralNetworkClassifier:
    """
    | NeuralNetworkClassifier depend on `Comet-ML <https://www.comet.ml/>`_ .
    | You have to create a project on your workspace of Comet, if you use this class.
    |
    | example

    ---------------------
    1st, Write your code.
    ---------------------
    ::

        # code.py
        from comet_ml import Experiment
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from SAnD.utils.trainer import NeuralNetworkClassifier

        class Network(nn.Module):
           def __init__(self):
               super(Network ,self).__init__()
               ...
           def forward(self, x):
               ...

        optimizer_config = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08}
        comet_config = {}

        train_val_loader = {
           "train": train_loader,
           "val": val_loader
        }
        test_loader = DataLoader(test_ds, batch_size)

        clf = NeuralNetworkClassifier(
                Network(), nn.CrossEntropyLoss(),
                optim.Adam, optimizer_config, Experiment()
            )

        clf.experiment_tag = "experiment_tag"
        clf.num_classes = 3
        clf.fit(train_val_loader, epochs=10)
        clf.evaluate(test_loader)
        lf.confusion_matrix(test_ds)
        clf.save_weights("save_params_test/")

    ----------------------------
    2nd, Run code on your shell.
    ----------------------------
    | You need to define 2 environment variables.
    | :code:`COMET_API_KEY` & :code:`COMET_PROJECT_NAME`

    On Unix-like system, you can define them like this and execute code.
    ::

        export COMET_API_KEY="YOUR-API-KEY"
        export COMET_PROJECT_NAME="YOUR-PROJECT-NAME"
        user@user$ python code.py

    -------------------------------------------
    3rd, check logs on your workspace of comet.
    -------------------------------------------
    Just access your `Comet-ML <https://www.comet.ml/>`_ Project page.

    ^^^^^
    Note,
    ^^^^^

    Execute this command on your shell, ::

        export COMET_DISABLE_AUTO_LOGGING=1

    If the following error occurs. ::

        ImportError: You must import Comet before these modules: torch

    """

    def __init__(self, model_s, model_n, model_ni, model_narx, criterion_s, criterion_n, criterion_ni, criterion_narx, optimizer, optimizer_config: dict, experiment) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # Si es 'cuda', entonces el entrenamiento se ejecutará en la GPU
        print("Device used for training:", self.device)
        self.model_s = model_s.to(self.device)
        self.model_n = model_n.to(self.device)
        self.model_ni = model_ni.to(self.device)
        self.model_narx = model_narx.to(self.device)
        self.optimizer_s = optimizer(self.model_s.parameters(), **optimizer_config)
        self.optimizer_n = optimizer(self.model_n.parameters(), **optimizer_config)
        self.optimizer_ni = optimizer(self.model_ni.parameters(), **optimizer_config)
        self.optimizer_narx = optimizer(self.model_narx.parameters(), **optimizer_config)
        self.criterion_s = criterion_s
        self.criterion_n = criterion_n
        self.criterion_ni = criterion_ni
        self.criterion_narx = criterion_narx
        self.experiment = experiment

        self.hyper_params = optimizer_config
        self._start_epoch = 0
        self.hyper_params["epochs"] = self._start_epoch
        self.__num_classes = None
        self._is_parallel = False

        # Suponiendo que tienes un modelo ya definido llamado `model`
        optimizer = optim.AdamW(model_ni.parameters(), lr=1e-3)


        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        #     self._is_parallel = True

        #     notice = "Running on {} GPUs.".format(torch.cuda.device_count())
        #     print("\033[33m" + notice + "\033[0m")

    def fit_siamese(self, x_train, y_train, x_val, y_val, x_test, y_test, loader: Dict[str, DataLoader], epochs: int, checkpoint_path: str = None, validation: bool = True) -> None:
        """
        | The method of training your PyTorch Model.
        | With the assumption, This method use for training network for classification.

        ::

            train_ds = Subset(train_val_ds, train_index)
            val_ds = Subset(train_val_ds, val_index)

            train_val_loader = {
                "train": DataLoader(train_ds, batch_size),
                "val": DataLoader(val_ds, batch_size)
            }

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.fit(train_val_loader, epochs=10)


        :param loader: Dictionary which contains Data Loaders for training and validation.: dict{DataLoader, DataLoader}
        :param epochs: The number of epochs: int
        :param checkpoint_path: str
        :param validation:
        :return: None
        """
        len_of_train_dataset = len(loader["train"].dataset)
        epochs = epochs + self._start_epoch

        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset

        if validation:
            len_of_val_dataset = len(loader["val"].dataset)
            self.hyper_params["val_ds_size"] = len_of_val_dataset

        self.experiment.log_parameters(self.hyper_params)

        for epoch in range(self._start_epoch, epochs):
            if checkpoint_path is not None and epoch % 100 == 0:
                self.save_to_file(checkpoint_path)
            with self.experiment.train():
                correct = 0.0
                total_loss = 0.0
                total_samples = 0.0

                self.model_s.train()
                pbar = tqdm.tqdm(total=len_of_train_dataset)

                for x1, y in loader["train"]:  # Ahora tenemos dos inputs + labels
                    x1_cont, x2_cont, y_cont = generar_pares_aleatorios(x_train, y_train, umbral_soh=0.02)
                    #for x1, x2, y in loader["train"]:  # Ahora tenemos dos inputs + labels
                    b_size = y.shape[0]
                    total_samples += y.shape[0]
                    x1_cont = x1_cont.to(self.device)if isinstance(x1_cont, torch.Tensor) else [i.to(self.device) for i in x1_cont]
                    x2_cont = x2_cont.to(self.device)if isinstance(x2_cont, torch.Tensor) else [i.to(self.device) for i in x2_cont]
                    #y_cont = y_cont.to(self.device)

                    pbar.set_description(
                        "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                    )
                    pbar.update(b_size)

                    # Forward pass (obtenemos los embeddings)
                    similarity = self.model_s(x1_cont, x2_cont)

                    #outputs = self.model(x)
                    # Calcular pérdida contrastiva
                    loss = self.criterion_s(similarity, y_cont)
                    self.optimizer_s.zero_grad()
                    loss.backward()
                    self.optimizer_s.step()

                    # Actualizar métricas
                    total_loss += loss.cpu().item()
                    avg_loss = total_loss / total_samples

                    # Registrar métricas en Comet o donde sea necesario
                    self.experiment.log_metric("loss", loss.cpu().item(), step=epoch)
                    self.experiment.log_metric("avg_loss", avg_loss, step=epoch)

                    # Registrar distancia media entre pares (métrica clave en aprendizaje siamés)
                    # avg_distance = torch.nn.functional.pairwise_distance(emb1, emb2).mean().cpu().item()
                    # self.experiment.log_metric("avg_embedding_distance", avg_distance, step=epoch)
            if validation:
                with self.experiment.validate():
                    with torch.no_grad():
                        val_correct = 0.0
                        val_total = 0.0

                        self.model_s.eval()
                        for x1_validation, y_validation in loader["val"]:
                            x1_cont_val, x2_cont_val, y_cont_val = generar_pares_aleatorios(x_val, y_val, umbral_soh=0.02)
                            val_total += y_validation.shape[0]
                            x1_cont_val = x1_cont_val.to(self.device) if isinstance(x1_cont_val, torch.Tensor) else [i_val.to(self.device) for i_val in x1_cont_val]
                            x2_cont_val = x2_cont_val.to(self.device) if isinstance(x2_cont_val, torch.Tensor) else [i_val.to(self.device) for i_val in x2_cont_val]
                            #y_val = y_val.to(self.device)

                            # Forward pass (obtenemos los embeddings)
                            similarity_val = self.model_s(x1_cont_val, x2_cont_val)
                            # Calcular pérdida contrastiva
                            val_loss = self.criterion_s(similarity_val, y_cont_val)
                            #val_output = self.model(x_val)
                            #val_loss = self.criterion(val_output, y_val)
                            # Calcular distancia entre embeddings
                            # distance = torch.nn.functional.pairwise_distance(emb1_val, emb2_val).mean().cpu().item()
                            # avg_distance += distance

                            # Acumular pérdida
                            total_loss += val_loss.cpu().item()

                            # Registrar métricas en Comet o donde sea necesario
                            self.experiment.log_metric("val_loss", val_loss.cpu().item(), step=epoch)
                            self.experiment.log_metric("avg_val_loss", total_loss / val_total, step=epoch)
                            # self.experiment.log_metric("avg_val_embedding_distance", avg_distance / total_samples, step=epoch)
            with self.experiment.test():
                running_loss = 0.0
                running_corrects = 0.0
                test_total = 0.0
                with torch.no_grad():
                    for x1_testing, y_testing in loader["test"]:
                        x1_cont_test, x2_cont_test, y_cont_test = generar_pares_aleatorios(x_test, y_test, umbral_soh=0.02)
                        test_total += y_testing.shape[0]
                        x1_cont_test = x1_cont_test.to(self.device) if isinstance(x1_cont_test, torch.Tensor) else [i_val.to(self.device) for i_val in x1_cont_test]
                        x2_cont_test = x2_cont_test.to(self.device) if isinstance(x2_cont_test, torch.Tensor) else [i_val.to(self.device) for i_val in x2_cont_test]
                        #y_test = y_test.to(self.device)
                        # x=y[0]
                        # y=y[1]
                        # #x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                        # y = y.to(self.device)
                        
                        pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
                        pbar.update(b_size)
                        # Forward pass (obtenemos los embeddings)
                        similarity_test = self.model_s(x1_cont_test, x2_cont_test)
                        # Calcular pérdida contrastiva
                        test_loss = self.criterion_s(similarity_test, y_cont_test)
                        # Calcular distancia entre embeddings
                        # distance = torch.nn.functional.pairwise_distance(emb1_test, emb2_test).mean().cpu().item()
                        # avg_distance += distance

                        # Acumular pérdida
                        running_corrects += test_loss.cpu().item()

                        self.experiment.log_metric("loss", running_corrects, step=epoch)
                        self.experiment.log_metric("accuracy", float(running_corrects / test_total))
                    pbar.close()
                    #acc = self.experiment.get_metric("accuracy")

            pbar.close()

    def fit_normal(self,x_train, y_train, x_val, y_val, x_test, y_test, loader: Dict[str, DataLoader], epochs: int, checkpoint_path: str = None, validation: bool = True, test: bool = True) -> None:
        """
        | The method of training your PyTorch Model.
        | With the assumption, This method use for training network for classification.

        ::

            train_ds = Subset(train_val_ds, train_index)
            val_ds = Subset(train_val_ds, val_index)

            train_val_loader = {
                "train": DataLoader(train_ds, batch_size),
                "val": DataLoader(val_ds, batch_size)
            }

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.fit(train_val_loader, epochs=10)


        :param loader: Dictionary which contains Data Loaders for training and validation.: dict{DataLoader, DataLoader}
        :param epochs: The number of epochs: int
        :param checkpoint_path: str
        :param validation:
        :return: None
        """
        len_of_train_dataset = len(loader["train"].dataset)
        epochs = epochs + self._start_epoch

        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset

        if validation:
            len_of_val_dataset = len(loader["val"].dataset)
            self.hyper_params["val_ds_size"] = len_of_val_dataset

        if test:
            len_of_test_dataset = len(loader["test"].dataset)
            self.hyper_params["test_ds_size"] = len_of_val_dataset

        self.experiment.log_parameters(self.hyper_params)

        for epoch in range(self._start_epoch, epochs):
            if checkpoint_path is not None and epoch % 100 == 0:
                self.save_to_file_normal(checkpoint_path)
            with self.experiment.train():
                train_correct = 0.0
                total_loss = 0.0
                total_samples = 0.0

                self.model_n.train()
                pbar = tqdm(total=len_of_train_dataset)
                for x_train, y_train in loader["train"]:
                    b_size = y_train.shape[0]
                    total_samples += y_train.shape[0]
                    x_train = x_train.to(self.device) if isinstance(x_train, torch.Tensor) else [i_val.to(self.device) for i_val in x_train]
                    y_train = y_train.to(self.device)
                    pbar.set_description(
                        "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                    )
                    pbar.update(b_size)
                    self.optimizer_n.zero_grad()
                    train_output = self.model_n(x_train)
                    train_loss = self.criterion_n(train_output, y_train)
                    train_loss.backward()
                    self.optimizer_n.step()
                    _, train_pred = torch.max(train_output, 1)
                    #val_correct += (val_pred == y_val).sum().float().item()
                    train_correct += (train_pred.to(self.device) == y_train.to(self.device)).sum().float().item()

                    self.experiment.log_metric("loss", train_loss.item(), step=epoch)
                    self.experiment.log_metric("accuracy", float(train_correct / total_samples), step=epoch)

                    # Actualizar métricas
                    total_loss += train_loss.item()
                    avg_loss = total_loss / total_samples

                    # Registrar métricas en Comet o donde sea necesario
                    #self.experiment.log_metric("loss", avg_loss.item(), step=epoch)
                    #self.experiment.log_metric("loss", float(avg_loss), step=epoch)
                    # self.experiment.log_metric("avg_loss", avg_loss, step=epoch)

                    # Registrar distancia media entre pares (métrica clave en aprendizaje siamés)
                    #avg_distance = torch.nn.functional.pairwise_distance(emb1, emb2).mean().item()
                    # self.experiment.log_metric("avg_embedding_distance", avg_distance, step=epoch)
            if validation:
                len_of_val_dataset = len(loader["val"].dataset)
                with self.experiment.validate():
                    with torch.no_grad():
                        val_correct = 0.0
                        val_total = 0.0

                        self.model_n.eval()
                        pbar = tqdm(total=len_of_val_dataset)
                        for x_val, y_val in loader["val"]:
                            b_size = y_val.shape[0]
                            val_total += y_val.shape[0]
                            x_val = x_val.to(self.device) if isinstance(x_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_val]
                            y_val = y_val.to(self.device)

                            pbar.set_description(
                                "\033[36m" + "Validating" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                            )
                            pbar.update(b_size)

                            val_output = self.model_n(x_val)
                            val_loss = self.criterion_n(val_output, y_val)
                            _, val_pred = torch.max(val_output, 1)
                            val_correct += (val_pred == y_val).sum().float().item()

                            # self.experiment.log_metric("loss", val_loss.item(), step=epoch)
                            # self.experiment.log_metric("accuracy", float(val_correct / val_total), step=epoch)

            if test:
                len_of_test_dataset = len(loader["test"].dataset)
                with self.experiment.test():
                    running_loss = 0.0
                    running_corrects = 0.0
                    with torch.no_grad():
                        test_correct = 0.0
                        test_total = 0.0
                        self.model_n.eval()
                        pbar = tqdm(total=len_of_test_dataset)
                        for x_test, y_test in loader["test"]:
                            b_size = y_test.shape[0]
                            test_total += y_test.shape[0]
                            x_test = x_test.to(self.device) if isinstance(x_test, torch.Tensor) else [i_val.to(self.device) for i_val in x_test]
                            y_test = y_test.to(self.device)
                            # x=y[0]
                            # y=y[1]
                            # #x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                            # y = y.to(self.device)

                            pbar.set_description(
                                "\033[36m" + "Testing" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                            )
                            pbar.update(b_size)
                            test_outputs = self.model_n(x_test)
                            test_loss = self.criterion_n(test_outputs, y_test)
                            _, test_predicted = torch.max(test_outputs, 1)
                            test_correct += (test_predicted == y_test).sum().float().item()

                            running_loss += test_loss.item()
                            running_corrects += torch.sum(test_predicted == y_test).float().item()

                            self.experiment.log_metric("loss", running_loss, step=epoch)
                            self.experiment.log_metric("accuracy", float(running_corrects / test_total))
                            # self.experiment.log_metric("predicted_soh", test_outputs.item(), step=epoch)
                            # self.experiment.log_metric("current_soh", x_test.item(), step=epoch)
                        pbar.close()
                        # acc = self.experiment.get_metric("accuracy")

            pbar.close()

    def fit_normal_improve(self,x_train, y_train, x_val, y_val, x_test, y_test, loader: Dict[str, DataLoader], epochs: int, checkpoint_path: str = None, validation: bool = True, test: bool = True) -> None:
        """
        | The method of training your PyTorch Model.
        | With the assumption, This method use for training network for classification.

        ::

            train_ds = Subset(train_val_ds, train_index)
            val_ds = Subset(train_val_ds, val_index)

            train_val_loader = {
                "train": DataLoader(train_ds, batch_size),
                "val": DataLoader(val_ds, batch_size)
            }

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.fit(train_val_loader, epochs=10)


        :param loader: Dictionary which contains Data Loaders for training and validation.: dict{DataLoader, DataLoader}
        :param epochs: The number of epochs: int
        :param checkpoint_path: str
        :param validation:
        :return: None
        """


        # Definir el scheduler StepLR: reduce el learning rate cada 10 epochs por un factor de 0.1
        # scheduler = optim.lr_scheduler.StepLR(self.optimizer_ni, step_size=5, gamma=0.5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_ni, T_max=50, eta_min=1e-6)
        len_of_train_dataset = len(loader["train"].dataset)
        epochs = epochs + self._start_epoch

        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset

        if validation:
            len_of_val_dataset = len(loader["val"].dataset)
            self.hyper_params["val_ds_size"] = len_of_val_dataset

        if test:
            len_of_test_dataset = len(loader["test"].dataset)
            self.hyper_params["test_ds_size"] = len_of_test_dataset

        self.experiment.log_parameters(self.hyper_params)

        for epoch in range(self._start_epoch, epochs):
            total_samples = 0
            if checkpoint_path is not None and epoch % 100 == 0:
                self.save_to_file_normal_improve(checkpoint_path)
            with self.experiment.train():
                train_correct = 0.0
                total_loss = 0.0
                total_samples = 0.0

                self.model_ni.train()
                pbar = tqdm(total=len_of_train_dataset)
                for x_train, y_train in loader["train"]:
                    b_size = y_train.shape[0]
                    total_samples += y_train.shape[0]
                    x_train = x_train.to(self.device) if isinstance(x_train, torch.Tensor) else [i_val.to(self.device) for i_val in x_train]
                    y_train = y_train.to(self.device)
                    pbar.set_description(
                        "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                    )
                    pbar.update(b_size)
                    self.optimizer_ni.zero_grad()
                    train_output = self.model_ni(x_train)
                    train_loss = self.criterion_ni(train_output, y_train)
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_ni.parameters(), max_norm=1.0)
                    self.optimizer_ni.step()
                    # _, train_pred = torch.max(train_output, 1)
                    # #val_correct += (val_pred == y_val).sum().float().item()
                    # train_correct += (train_pred.to(self.device) == y_train.to(self.device)).sum().float().item()

                    # Predicciones continuas
                    train_pred = train_output

                    # Comparar las predicciones con las etiquetas reales usando una métrica de error
                    #train_loss = torch.nn.functional.mse_loss(train_pred, y_train.to(self.device))
                    train_loss = torch.nn.functional.mse_loss(train_pred, y_train.unsqueeze(1).to(self.device))


                    # Si quieres llevar un conteo de cuántas predicciones están cerca del valor real (por ejemplo, dentro de un umbral)
                    threshold = 0.1  # Definir un umbral de tolerancia para considerarlo "correcto"
                    correct_preds = ((train_pred - y_train.to(self.device)).abs() < threshold).sum().float().item()
                    train_correct += correct_preds

                    #total_samples = 24950
                    self.experiment.log_metric("loss", train_loss.item(), step=epoch)
                    self.experiment.log_metric("accuracy", float(train_correct / total_samples), step=epoch)

                    # Actualizar métricas
                    total_loss += train_loss.item()
                    avg_loss = total_loss / total_samples

                    # Registrar métricas en Comet o donde sea necesario
                    #self.experiment.log_metric("loss", avg_loss.item(), step=epoch)
                    #self.experiment.log_metric("loss", float(avg_loss), step=epoch)
                    # self.experiment.log_metric("avg_loss", avg_loss, step=epoch)

                    # Registrar distancia media entre pares (métrica clave en aprendizaje siamés)
                    #avg_distance = torch.nn.functional.pairwise_distance(emb1, emb2).mean().item()
                    # self.experiment.log_metric("avg_embedding_distance", avg_distance, step=epoch)
            if validation:
                len_of_val_dataset = len(loader["val"].dataset)
                with self.experiment.validate():
                    with torch.no_grad():
                        val_correct = 0.0
                        val_total = 0.0

                        self.model_ni.eval()
                        pbar = tqdm(total=len_of_val_dataset)
                        for x_val, y_val in loader["val"]:
                            b_size = y_val.shape[0]
                            val_total += y_val.shape[0]
                            x_val = x_val.to(self.device) if isinstance(x_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_val]
                            y_val = y_val.to(self.device)

                            pbar.set_description(
                                "\033[36m" + "Validating" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                            )
                            pbar.update(b_size)

                            val_output = self.model_ni(x_val)
                            val_loss = self.criterion_ni(val_output, y_val)
                            _, val_pred = torch.max(val_output, 1)
                            val_correct += (val_pred == y_val).sum().float().item()

                            # self.experiment.log_metric("loss", val_loss.item(), step=epoch)
                            # self.experiment.log_metric("accuracy", float(val_correct / val_total), step=epoch)
            # Paso del scheduler después de cada epoch
            scheduler.step()
            # Mostrar el learning rate actual
            print(f'Epoch [{epoch+1}/{epochs}], Learning Rate: {scheduler.get_last_lr()[0]}')

            if test:
                len_of_test_dataset = len(loader["test"].dataset)
                with self.experiment.test():
                    running_loss = 0.0
                    running_corrects = 0.0
                    with torch.no_grad():
                        test_correct = 0.0
                        test_total = 0.0
                        self.model_ni.eval()
                        pbar = tqdm(total=len_of_test_dataset)
                        for x_test, y_test in loader["test"]:
                            b_size = y_test.shape[0]
                            test_total += y_test.shape[0]
                            x_test = x_test.to(self.device) if isinstance(x_test, torch.Tensor) else [i_val.to(self.device) for i_val in x_test]
                            y_test = y_test.to(self.device)
                            # x=y[0]
                            # y=y[1]
                            # #x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                            # y = y.to(self.device)

                            pbar.set_description(
                                "\033[36m" + "Testing" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                            )
                            pbar.update(b_size)
                            test_outputs = self.model_ni(x_test)
                            # test_loss = self.criterion_ni(test_outputs, y_test)
                            # _, test_predicted = torch.max(test_outputs, 1)
                            # test_correct += (test_predicted.to(self.device) == y_test.to(self.device)).sum().float().item()
                            # running_corrects += torch.sum(test_predicted == y_test).float().item()
                            #
                            # self.experiment.log_metric("loss", test_loss, step=epoch)
                            # self.experiment.log_metric("accuracy", float(running_corrects / test_total))


                            # Predicciones continuas
                            test_predicted = test_outputs
                            # Comparar las predicciones con las etiquetas reales usando una métrica de error
                            test_loss = torch.nn.functional.mse_loss(test_predicted, y_test.to(self.device))
                            # Si quieres llevar un conteo de cuántas predicciones están cerca del valor real (por ejemplo, dentro de un umbral)
                            threshold = 0.1  # Definir un umbral de tolerancia para considerarlo "correcto"
                            test_correct += ((test_predicted - y_test.to(self.device)).abs() < threshold).sum().float().item()

                            self.experiment.log_metric("loss", test_loss.item(), step=epoch)
                            self.experiment.log_metric("accuracy", float(test_correct / total_samples), step=epoch)


                            # self.experiment.log_metric("predicted_soh", test_outputs.item(), step=epoch)
                            # self.experiment.log_metric("current_soh", x_test.item(), step=epoch)
                        pbar.close()
                        # acc = self.experiment.get_metric("accuracy")

            pbar.close()



    def fit_NARX_Transformer(self,x_train, y_train, x_val, y_val, x_test, y_test, loader: Dict[str, DataLoader], epochs: int, checkpoint_path: str = None, validation: bool = True, test: bool = True) -> None:
        # Loss function and optimizer
        """
        | The method of training your PyTorch Model.
        | With the assumption, This method use for training network for classification.

        ::

            train_ds = Subset(train_val_ds, train_index)
            val_ds = Subset(train_val_ds, val_index)

            train_val_loader = {
                "train": DataLoader(train_ds, batch_size),
                "val": DataLoader(val_ds, batch_size)
            }

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.fit(train_val_loader, epochs=10)


        :param loader: Dictionary which contains Data Loaders for training and validation.: dict{DataLoader, DataLoader}
        :param epochs: The number of epochs: int
        :param checkpoint_path: str
        :param validation:
        :return: None
        """


        # Definir el scheduler StepLR: reduce el learning rate cada 10 epochs por un factor de 0.1
        # scheduler = optim.lr_scheduler.StepLR(self.optimizer_ni, step_size=5, gamma=0.5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_narx, T_max=50, eta_min=1e-6)
        len_of_train_dataset = len(loader["train_narx"].dataset)
        epochs = epochs + self._start_epoch

        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train_narx"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset

        if validation:
            len_of_val_dataset = len(loader["val_narx"].dataset)
            self.hyper_params["val_ds_size"] = len_of_val_dataset

        if test:
            len_of_test_dataset = len(loader["test_narx"].dataset)
            self.hyper_params["test_ds_size"] = len_of_test_dataset

        self.experiment.log_parameters(self.hyper_params)

        for epoch in range(self._start_epoch, epochs):
            total_samples = 0
            if checkpoint_path is not None and epoch % 100 == 0:
                self.save_to_file_normal_improve(checkpoint_path)
            with self.experiment.train():
                train_correct = 0.0
                total_loss = 0.0
                total_samples = 0.0

                self.model_narx.train()
                pbar = tqdm(total=len_of_train_dataset)
                # for data in loader["train_narx"]:
                #     print(data)
                #     break  # Para ver solo el primer lote
                for x_train_narx, cap_train, y_train_narx in loader["train_narx"]:
                    b_size = y_train_narx.shape[0]
                    total_samples += y_train_narx.shape[0]
                    x_train_narx = x_train_narx.to(self.device)  # (batch_size, 2, 400, 3)
                    cap_train = cap_train.to(self.device)  # (batch_size, 1)
                    y_train_narx = y_train_narx.to(self.device)    # (batch_size)

                    pbar.set_description("\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs))
                    pbar.update(b_size)
                    self.optimizer_narx.zero_grad()
                    train_output = self.model_narx(x_train_narx, cap_train)
                    train_loss = self.criterion_narx(train_output, y_train_narx.unsqueeze(1))
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_narx.parameters(), max_norm=1.0)
                    self.optimizer_narx.step()
                    # _, train_pred = torch.max(train_output, 1)
                    # #val_correct += (val_pred == y_val).sum().float().item()
                    # train_correct += (train_pred.to(self.device) == y_train.to(self.device)).sum().float().item()

                    # Predicciones continuas
                    train_pred = train_output

                    # Comparar las predicciones con las etiquetas reales usando una métrica de error
                    #train_loss = torch.nn.functional.mse_loss(train_pred, y_train.to(self.device))
                    # train_loss = torch.nn.functional.mse_loss(train_pred, y_train.unsqueeze(1).to(self.device))


                    # Si quieres llevar un conteo de cuántas predicciones están cerca del valor real (por ejemplo, dentro de un umbral)
                    threshold = 0.1  # Definir un umbral de tolerancia para considerarlo "correcto"
                    correct_preds = ((train_pred - y_train.to(self.device)).abs() < threshold).sum().float().item()
                    train_correct += correct_preds

                    total_samples = 24950
                    self.experiment.log_metric("loss", train_loss.item(), step=epoch)
                    self.experiment.log_metric("accuracy", float(train_correct / total_samples), step=epoch)

                    # Actualizar métricas
                    total_loss += train_loss.item()
                    avg_loss = total_loss / total_samples

                    # Registrar métricas en Comet o donde sea necesario
                    #self.experiment.log_metric("loss", avg_loss.item(), step=epoch)
                    #self.experiment.log_metric("loss", float(avg_loss), step=epoch)
                    # self.experiment.log_metric("avg_loss", avg_loss, step=epoch)

                    # Registrar distancia media entre pares (métrica clave en aprendizaje siamés)
                    #avg_distance = torch.nn.functional.pairwise_distance(emb1, emb2).mean().item()
                    # self.experiment.log_metric("avg_embedding_distance", avg_distance, step=epoch)
            if validation:
                len_of_val_dataset = len(loader["val_narx"].dataset)
                with self.experiment.validate():
                    with torch.no_grad():
                        val_correct = 0.0
                        val_total = 0.0

                        self.model_narx.eval()
                        pbar = tqdm(total=len_of_val_dataset)
                        for x_val_narx, cap_val, y_val_narx in loader["val_narx"]:
                            b_size = y_val_narx.shape[0]
                            val_total += y_val_narx.shape[0]
                            x_val_narx = x_val_narx.to(self.device) if isinstance(x_val_narx, torch.Tensor) else [i_val.to(self.device) for i_val in x_val_narx]
                            y_val_narx = y_val_narx.to(self.device)
                            cap_val = cap_val.to(self.device)

                            pbar.set_description(
                                "\033[36m" + "Validating" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                            )
                            pbar.update(b_size)

                            val_output = self.model_narx(x_val_narx, cap_val)
                            val_loss = self.criterion_narx(val_output, y_val_narx)
                            _, val_pred = torch.max(val_output, 1)
                            val_correct += (val_pred == y_val_narx).sum().float().item()

                            # self.experiment.log_metric("loss", val_loss.item(), step=epoch)
                            # self.experiment.log_metric("accuracy", float(val_correct / val_total), step=epoch)
            # Paso del scheduler después de cada epoch
            scheduler.step()
            # Mostrar el learning rate actual
            print(f'Epoch [{epoch+1}/{epochs}], Learning Rate: {scheduler.get_last_lr()[0]}')

            if test:
                len_of_test_dataset = len(loader["test_narx"].dataset)
                with self.experiment.test():
                    running_loss = 0.0
                    running_corrects = 0.0
                    with torch.no_grad():
                        test_correct = 0.0
                        test_total = 0.0
                        self.model_narx.eval()
                        pbar = tqdm(total=len_of_test_dataset)
                        for x_test_narx, cap_test, y_test_narx  in loader["test_narx"]:
                            b_size = y_test_narx.shape[0]
                            test_total += y_test_narx.shape[0]
                            x_test_narx = x_test_narx.to(self.device) if isinstance(x_test_narx, torch.Tensor) else [i_val.to(self.device) for i_val in x_test_narx]
                            y_test_narx = y_test_narx.to(self.device)
                            cap_test = cap_test.to(self.device)
                            # x=y[0]
                            # y=y[1]
                            # #x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                            # y = y.to(self.device)

                            pbar.set_description(
                                "\033[36m" + "Testing" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                            )
                            pbar.update(b_size)
                            test_outputs = self.model_narx(x_test_narx, cap_test)
                            # test_loss = self.criterion_ni(test_outputs, y_test)
                            # _, test_predicted = torch.max(test_outputs, 1)
                            # test_correct += (test_predicted.to(self.device) == y_test.to(self.device)).sum().float().item()
                            # running_corrects += torch.sum(test_predicted == y_test).float().item()
                            #
                            # self.experiment.log_metric("loss", test_loss, step=epoch)
                            # self.experiment.log_metric("accuracy", float(running_corrects / test_total))


                            # Predicciones continuas
                            test_predicted = test_outputs
                            # Comparar las predicciones con las etiquetas reales usando una métrica de error
                            test_loss = torch.nn.functional.mse_loss(test_predicted, y_test_narx.to(self.device))
                            # Si quieres llevar un conteo de cuántas predicciones están cerca del valor real (por ejemplo, dentro de un umbral)
                            threshold = 0.1  # Definir un umbral de tolerancia para considerarlo "correcto"
                            test_correct += ((test_predicted - y_test_narx.to(self.device)).abs() < threshold).sum().float().item()

                            self.experiment.log_metric("loss", test_loss.item(), step=epoch)
                            self.experiment.log_metric("accuracy", float(test_correct / total_samples), step=epoch)


                            # self.experiment.log_metric("predicted_soh", test_outputs.item(), step=epoch)
                            # self.experiment.log_metric("current_soh", x_test.item(), step=epoch)
                        pbar.close()
                        # acc = self.experiment.get_metric("accuracy")

            pbar.close()


    def fit_NARX_Transformer_finetuning(self, epoch, optimizer, modelo, x_train, y_train, x_val, y_val, x_test, y_test, loader: Dict[str, DataLoader], epochs: int, checkpoint_path: str = None, validation: bool = True, test: bool = True) -> None:
        # Loss function and optimizer
        """
        | The method of training your PyTorch Model.
        | With the assumption, This method use for training network for classification.

        ::

            train_ds = Subset(train_val_ds, train_index)
            val_ds = Subset(train_val_ds, val_index)

            train_val_loader = {
                "train": DataLoader(train_ds, batch_size),
                "val": DataLoader(val_ds, batch_size)
            }

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.fit(train_val_loader, epochs=10)


        :param loader: Dictionary which contains Data Loaders for training and validation.: dict{DataLoader, DataLoader}
        :param epochs: The number of epochs: int
        :param checkpoint_path: str
        :param validation:
        :return: None
        """
        self.optimizer_narx = optimizer
        self._start_epoch = epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_narx, T_max=50, eta_min=1e-6)
        len_of_train_dataset = len(loader["train_narx"].dataset)
        epochs = epochs + self._start_epoch
        self.modelo = modelo.to(self.device)
        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train_narx"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset

        if validation:
            len_of_val_dataset = len(loader["val_narx"].dataset)
            self.hyper_params["val_ds_size"] = len_of_val_dataset

        if test:
            len_of_test_dataset = len(loader["test_narx"].dataset)
            self.hyper_params["test_ds_size"] = len_of_test_dataset

        self.experiment.log_parameters(self.hyper_params)

        for epoch in range(self._start_epoch, epochs):
            total_samples = 0
            if checkpoint_path is not None and epoch % 100 == 0:
                self.save_to_file_normal_improve(checkpoint_path)
            with self.experiment.train():
                train_correct = 0.0
                total_loss = 0.0
                total_samples = 0.0

                self.modelo.train()
                pbar = tqdm(total=len_of_train_dataset)
                # for data in loader["train_narx"]:
                #     print(data)
                #     break  # Para ver solo el primer lote
                for x_train_narx, cap_train, y_train_narx in loader["train_narx"]:
                    b_size = y_train_narx.shape[0]
                    total_samples += y_train_narx.shape[0]
                    x_train_narx = x_train_narx.to(self.device)  # (batch_size, 2, 400, 3)
                    cap_train = cap_train.to(self.device)  # (batch_size, 1)
                    y_train_narx = y_train_narx.to(self.device)    # (batch_size)

                    pbar.set_description("\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs))
                    pbar.update(b_size)
                    self.optimizer_narx.zero_grad()
                    train_output = self.modelo(x_train_narx, cap_train)
                    train_loss = self.criterion_narx(train_output, y_train_narx.unsqueeze(1))
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=1.0)
                    self.optimizer_narx.step()
                    # _, train_pred = torch.max(train_output, 1)
                    # #val_correct += (val_pred == y_val).sum().float().item()
                    # train_correct += (train_pred.to(self.device) == y_train.to(self.device)).sum().float().item()

                    # Predicciones continuas
                    train_pred = train_output

                    # Comparar las predicciones con las etiquetas reales usando una métrica de error
                    #train_loss = torch.nn.functional.mse_loss(train_pred, y_train.to(self.device))
                    # train_loss = torch.nn.functional.mse_loss(train_pred, y_train.unsqueeze(1).to(self.device))


                    # Si quieres llevar un conteo de cuántas predicciones están cerca del valor real (por ejemplo, dentro de un umbral)
                    threshold = 0.1  # Definir un umbral de tolerancia para considerarlo "correcto"
                    correct_preds = ((train_pred - y_train.to(self.device)).abs() < threshold).sum().float().item()
                    train_correct += correct_preds

                    total_samples = 24950
                    self.experiment.log_metric("loss", train_loss.item(), step=epoch)
                    self.experiment.log_metric("accuracy", float(train_correct / total_samples), step=epoch)

                    # Actualizar métricas
                    total_loss += train_loss.item()
                    avg_loss = total_loss / total_samples

                    # Registrar métricas en Comet o donde sea necesario
                    #self.experiment.log_metric("loss", avg_loss.item(), step=epoch)
                    #self.experiment.log_metric("loss", float(avg_loss), step=epoch)
                    # self.experiment.log_metric("avg_loss", avg_loss, step=epoch)

                    # Registrar distancia media entre pares (métrica clave en aprendizaje siamés)
                    #avg_distance = torch.nn.functional.pairwise_distance(emb1, emb2).mean().item()
                    # self.experiment.log_metric("avg_embedding_distance", avg_distance, step=epoch)
            if validation:
                len_of_val_dataset = len(loader["val_narx"].dataset)
                with self.experiment.validate():
                    with torch.no_grad():
                        val_correct = 0.0
                        val_total = 0.0

                        self.modelo.eval()
                        pbar = tqdm(total=len_of_val_dataset)
                        for x_val_narx, cap_val, y_val_narx in loader["val_narx"]:
                            b_size = y_val_narx.shape[0]
                            val_total += y_val_narx.shape[0]
                            x_val_narx = x_val_narx.to(self.device) if isinstance(x_val_narx, torch.Tensor) else [i_val.to(self.device) for i_val in x_val_narx]
                            y_val_narx = y_val_narx.to(self.device)
                            cap_val = cap_val.to(self.device)

                            pbar.set_description(
                                "\033[36m" + "Validating" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                            )
                            pbar.update(b_size)

                            val_output = self.modelo(x_val_narx, cap_val)
                            val_loss = self.criterion_narx(val_output, y_val_narx)
                            _, val_pred = torch.max(val_output, 1)
                            val_correct += (val_pred == y_val_narx).sum().float().item()

                            # self.experiment.log_metric("loss", val_loss.item(), step=epoch)
                            # self.experiment.log_metric("accuracy", float(val_correct / val_total), step=epoch)
            # Paso del scheduler después de cada epoch
            scheduler.step()
            # Mostrar el learning rate actual
            print(f'Epoch [{epoch+1}/{epochs}], Learning Rate: {scheduler.get_last_lr()[0]}')

            if test:
                len_of_test_dataset = len(loader["test_narx"].dataset)
                with self.experiment.test():
                    running_loss = 0.0
                    running_corrects = 0.0
                    with torch.no_grad():
                        test_correct = 0.0
                        test_total = 0.0
                        self.modelo.eval()
                        pbar = tqdm(total=len_of_test_dataset)
                        for x_test_narx, cap_test, y_test_narx  in loader["test_narx"]:
                            b_size = y_test_narx.shape[0]
                            test_total += y_test_narx.shape[0]
                            x_test_narx = x_test_narx.to(self.device) if isinstance(x_test_narx, torch.Tensor) else [i_val.to(self.device) for i_val in x_test_narx]
                            y_test_narx = y_test_narx.to(self.device)
                            cap_test = cap_test.to(self.device)
                            # x=y[0]
                            # y=y[1]
                            # #x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                            # y = y.to(self.device)

                            pbar.set_description(
                                "\033[36m" + "Testing" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                            )
                            pbar.update(b_size)
                            test_outputs = self.modelo(x_test_narx, cap_test)
                            # test_loss = self.criterion_ni(test_outputs, y_test)
                            # _, test_predicted = torch.max(test_outputs, 1)
                            # test_correct += (test_predicted.to(self.device) == y_test.to(self.device)).sum().float().item()
                            # running_corrects += torch.sum(test_predicted == y_test).float().item()
                            #
                            # self.experiment.log_metric("loss", test_loss, step=epoch)
                            # self.experiment.log_metric("accuracy", float(running_corrects / test_total))


                            # Predicciones continuas
                            test_predicted = test_outputs
                            # Comparar las predicciones con las etiquetas reales usando una métrica de error
                            test_loss = torch.nn.functional.mse_loss(test_predicted, y_test_narx.to(self.device))
                            # Si quieres llevar un conteo de cuántas predicciones están cerca del valor real (por ejemplo, dentro de un umbral)
                            threshold = 0.1  # Definir un umbral de tolerancia para considerarlo "correcto"
                            test_correct += ((test_predicted - y_test_narx.to(self.device)).abs() < threshold).sum().float().item()

                            self.experiment.log_metric("loss", test_loss.item(), step=epoch)
                            self.experiment.log_metric("accuracy", float(test_correct / total_samples), step=epoch)


                            # self.experiment.log_metric("predicted_soh", test_outputs.item(), step=epoch)
                            # self.experiment.log_metric("current_soh", x_test.item(), step=epoch)
                        pbar.close()
                        # acc = self.experiment.get_metric("accuracy")

            pbar.close()




    def evaluate(self, loader: DataLoader, verbose: bool = False) -> None or float:
        """
        The method of evaluating your PyTorch Model.
        With the assumption, This method use for training network for classification.

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.evaluate(test_loader)


        :param loader: DataLoader for Evaluating: torch.utils.data.DataLoader
        :param verbose: bool
        :return: None
        """
        running_loss = 0.0
        running_corrects = 0.0
        pbar = tqdm.tqdm(total=len(loader.dataset))


        self.model_n.eval()
        self.experiment.log_parameter("test_ds_size", len(loader.dataset))
        with self.experiment.test():
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                for x, y in enumerate(loader):
                    b_size = len(y)
                    total += len(y)
                    x=y[0]
                    y=y[1]
                    #x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                    y = y.to(self.device)
                    
                    pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
                    pbar.update(b_size)

                    outputs = self.model_n(x)
                    loss = self.criterion_n(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().float().cpu().item()

                    running_loss += loss.cpu().item()
                    running_corrects += torch.sum(predicted == y).float().cpu().item()

                    self.experiment.log_metric("loss", running_loss)
                    self.experiment.log_metric("accuracy", float(running_corrects / total))
                pbar.close()
            #acc = self.experiment.get_metric("accuracy")

        print("\033[33m" + "Evaluation finished. " + "\033[0m" + "Loss: {:.4f}".format(loss))

        if verbose:
            return loss

    def save_checkpoint_n_improve(self) -> dict:
        """
        The method of saving trained PyTorch model.

        Note,  return value contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            checkpoints = clf.save_checkpoint()

        :return: dict {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        """

        checkpoints = {
            "epoch": deepcopy(self.hyper_params["epochs"]),
            "optimizer_state_dict": deepcopy(self.optimizer_ni.state_dict())
        }

        if self._is_parallel:
            checkpoints["model_state_dict"] = deepcopy(self.model_ni.module.state_dict())
        else:
            checkpoints["model_state_dict"] = deepcopy(self.model_ni.state_dict())

        return checkpoints



    def save_checkpoint_narx(self) -> dict:
        """
        The method of saving trained PyTorch model.

        Note,  return value contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            checkpoints = clf.save_checkpoint()

        :return: dict {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        """

        checkpoints = {
            "epoch": deepcopy(self.hyper_params["epochs"]),
            "optimizer_state_dict": deepcopy(self.optimizer_narx.state_dict())
        }

        if self._is_parallel:
            checkpoints["model_state_dict"] = deepcopy(self.model_narx.module.state_dict())
        else:
            checkpoints["model_state_dict"] = deepcopy(self.model_narx.state_dict())

        return checkpoints


    def save_checkpoint_n(self) -> dict:
        """
        The method of saving trained PyTorch model.

        Note,  return value contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            checkpoints = clf.save_checkpoint()

        :return: dict {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        """

        checkpoints = {
            "epoch": deepcopy(self.hyper_params["epochs"]),
            "optimizer_state_dict": deepcopy(self.optimizer_n.state_dict())
        }

        if self._is_parallel:
            checkpoints["model_state_dict"] = deepcopy(self.model_n.module.state_dict())
        else:
            checkpoints["model_state_dict"] = deepcopy(self.model_n.state_dict())

        return checkpoints


    def save_checkpoint_s(self) -> dict:
        """
        The method of saving trained PyTorch model.

        Note,  return value contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            checkpoints = clf.save_checkpoint()

        :return: dict {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        """

        checkpoints = {
            "epoch": deepcopy(self.hyper_params["epochs"]),
            "optimizer_state_dict": deepcopy(self.optimizer_s.state_dict())
        }

        if self._is_parallel:
            checkpoints["model_state_dict"] = deepcopy(self.model_s.module.state_dict())
        else:
            checkpoints["model_state_dict"] = deepcopy(self.model_s.state_dict())

        return checkpoints

    def save_to_file_normal_improve(self, path: str) -> str:
        """
        | The method of saving trained PyTorch model to file.
        | Those weights are uploaded to comet.ml as backup.
        | check "Asserts".

        Note, .pth file contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            filename = clf.save_to_file('path/to/save/dir/')

        :param path: path to saving directory. : string
        :return: path to file : string
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        # file_name = "model_params-epochs_{}-{}.pth".format(
        #     self.hyper_params["epochs"], time.ctime().replace(" ", "_")
        # )
        file_name = "trained_model_normal_improve.pth"
        path = path + file_name

        checkpoints = self.save_checkpoint_n_improve()

        # torch.save(checkpoints, path,{"hyperparameters": hyperparameters})
        torch.save(checkpoints, path)
        self.experiment.log_asset(path, file_name=file_name)

        return path

    def save_to_file_normal(self, path: str) -> str:
        """
        | The method of saving trained PyTorch model to file.
        | Those weights are uploaded to comet.ml as backup.
        | check "Asserts".

        Note, .pth file contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            filename = clf.save_to_file('path/to/save/dir/')

        :param path: path to saving directory. : string
        :return: path to file : string
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        # file_name = "model_params-epochs_{}-{}.pth".format(
        #     self.hyper_params["epochs"], time.ctime().replace(" ", "_")
        # )
        file_name = "trained_model_normal.pth"
        path = path + file_name

        checkpoints = self.save_checkpoint_n()

        torch.save(checkpoints, path)
        self.experiment.log_asset(path, file_name=file_name)

        return path

    def save_to_file_normal_improve(self, path: str) -> str:
        """
        | The method of saving trained PyTorch model to file.
        | Those weights are uploaded to comet.ml as backup.
        | check "Asserts".

        Note, .pth file contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            filename = clf.save_to_file('path/to/save/dir/')

        :param path: path to saving directory. : string
        :return: path to file : string
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        # file_name = "model_params-epochs_{}-{}.pth".format(
        #     self.hyper_params["epochs"], time.ctime().replace(" ", "_")
        # )
        file_name = "trained_model_normal_improve.pth"
        path = path + file_name

        checkpoints = self.save_checkpoint_n_improve()
        torch.save(checkpoints, path)
        self.experiment.log_asset(path, file_name=file_name)

        return path


    def save_to_file_siamese(self, path: str) -> str:
        """
        | The method of saving trained PyTorch model to file.
        | Those weights are uploaded to comet.ml as backup.
        | check "Asserts".

        Note, .pth file contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            filename = clf.save_to_file('path/to/save/dir/')

        :param path: path to saving directory. : string
        :return: path to file : string
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        # file_name = "model_params-epochs_{}-{}.pth".format(
        #     self.hyper_params["epochs"], time.ctime().replace(" ", "_")
        # )
        file_name = "trained_model_siamese.pth"
        path = path + file_name

        checkpoints = self.save_checkpoint_s()

        torch.save(checkpoints, path)
        self.experiment.log_asset(path, file_name=file_name)

        return path


    def save_to_file_Narx(self, path: str) -> str:
        """
        | The method of saving trained PyTorch model to file.
        | Those weights are uploaded to comet.ml as backup.
        | check "Asserts".

        Note, .pth file contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            filename = clf.save_to_file('path/to/save/dir/')

        :param path: path to saving directory. : string
        :return: path to file : string
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        # file_name = "model_params-epochs_{}-{}.pth".format(
        #     self.hyper_params["epochs"], time.ctime().replace(" ", "_")
        # )
        file_name = "trained_model_narx_2var.pth"
        path = path + file_name

        checkpoints = self.save_checkpoint_narx()

        # torch.save(checkpoints, path,{"hyperparameters": hyperparameters})
        torch.save(checkpoints, path)
        self.experiment.log_asset(path, file_name=file_name)

        return path


    def save_to_file_Narx_finetuning(self, file_name: str) -> str:
        """
        | The method of saving trained PyTorch model to file.
        | Those weights are uploaded to comet.ml as backup.
        | check "Asserts".

        Note, .pth file contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            filename = clf.save_to_file('path/to/save/dir/')

        :param path: path to saving directory. : string
        :return: path to file : string
        """
        path = "save_params/"
        if not os.path.isdir(path):
            os.mkdir(path)

        # file_name = "model_params-epochs_{}-{}.pth".format(
        #     self.hyper_params["epochs"], time.ctime().replace(" ", "_")
        # )

        checkpoints = self.save_checkpoint_narx()
        path = path + file_name

        # torch.save(checkpoints, path,{"hyperparameters": hyperparameters})
        torch.save(checkpoints, path)
        self.experiment.log_asset(path, file_name=file_name)

        return path



    def restore_checkpoint(self, checkpoints: dict) -> None:
        """
        The method of loading trained PyTorch model.

        :param checkpoints: dictionary which contains {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        :return: None
        """
        self._start_epoch = checkpoints["epoch"]
        if not isinstance(self._start_epoch, int):
            raise TypeError

        if self._is_parallel:
            self.model_n.module.load_state_dict(checkpoints["model_state_dict"])
        else:
            self.model_n.load_state_dict(checkpoints["model_state_dict"])

        self.optimizer_n.load_state_dict(checkpoints["optimizer_state_dict"])

    def restore_from_file(self, path: str, map_location: str = "cpu") -> None:
        """
        The method of loading trained PyTorch model from file.

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.restore_from_file('path/to/trained/weights.pth')

        :param path: path to saved directory. : str
        :param map_location: default cpu: str
        :return: None
        """
        checkpoints = torch.load(path, map_location=map_location)
        self.restore_checkpoint(checkpoints)

    @property
    def experiment_tag(self) -> list:
        return self.experiment.get_tags()

    @experiment_tag.setter
    def experiment_tag(self, tag: str) -> None:
        """
        ::

            clf = NeuralNetworkClassifier(...)
            clf.experiment_tag = "tag"

        :param tag: str
        :return: None
        """
        if not isinstance(tag, str):
            raise TypeError

        self.experiment.add_tag(tag)

    @property
    def num_class(self) -> int or None:
        return self.__num_classes

    @num_class.setter
    def num_class(self, num_class: int) -> None:
        if not (isinstance(num_class, int) and num_class > 0):
            raise Exception("the number of class must be greater than 0.")

        self.__num_classes = num_class
        self.experiment.log_parameter("classes", self.__num_classes)

    def confusion_matrix(self, dataset: torch.utils.data.Dataset, labels=None, sample_weight=None) -> None:
        """
        | Generate confusion matrix.
        | result save on comet.ml.

        :param dataset: dataset for generating confusion matrix.
        :param labels: array, shape = [n_samples]
        :param sample_weight: array-lie of shape = [n_samples], optional
        :return: None
        """
        targets = []
        predicts = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        pbar = tqdm.tqdm(total=len(loader.dataset))

        self.model.eval()

        with torch.no_grad():
            for step, (x, y) in enumerate(loader):
                x = x.to(self.device)

                pbar.set_description("\033[31m" + "Calculating confusion matrix" + "\033[0m")
                pbar.update(step)

                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)

                predicts.append(predicted.cpu().numpy())
                targets.append(y.numpy())
            pbar.close()

        cm = pd.DataFrame(confusion_matrix(targets, predicts, labels, sample_weight))
        self.experiment.log_asset_data(
            cm.to_csv(), "ConfusionMatrix-epochs-{}-{}.csv".format(
                self.hyper_params["epochs"], time.ctime().replace(" ", "_")
            )
        )

