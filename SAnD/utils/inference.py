import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sympy import false
from torch.utils.data import DataLoader
import yaml

from dataset import load_NASA

from SAnD.core.model import SAnD, SAnD_Embedding, SiameseSAnD, SAnDImprove, NARX_Transformer, NARX_Transformer_2var, NARX_Transformer_2var_SoloActual


class Inference_SoH_Siamese:
    def __init__(self, model_path, input_features, seq_len, n_heads, factor, n_class, n_layers, device="cuda"):
        self.device = device
        self.siamese_model = SAnD_Embedding(input_features, seq_len, n_heads, factor, n_class, n_layers)
        checkpoint = torch.load("save_params/trained_model_siamese.pth", map_location=device)
        self.siamese_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        #self.siamese_model.load_state_dict(torch.load(model_path, map_location=device))
        self.siamese_model.to(device)
        self.siamese_model.eval()
        # self.sand_model = SAnD(input_features, seq_len, n_heads, factor, n_class, n_layers)
        # self.siamese_model.encoder.load_state_dict(self.siamese_model.sand.encoder.state_dict())
        self.siamese_model.to(device)
        self.siamese_model.eval()

    def predict(self, test_loader):
        predictions = []
        soh_real = []
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.clone().detach().to(self.device)
                x_test = x_test.to(self.device)
                soh_raw = self.siamese_model(x_test)  # Obtener SoH
                soh_pred = soh_raw.cpu().numpy()  # Mover a CPU y convertir a NumPy
                predictions.append(soh_pred)
                soh_real.append(y_test.cpu().numpy())
            # ##########################################################################
        # # PLoteo de la inferencia
        # Convertir listas a numpy arrays
        soh_pred = np.concatenate(predictions).flatten()
        soh_real = np.concatenate(soh_real).flatten()

        # Llamar a la función de visualización
        # self.plot_soh(soh_real, soh_pred)

        # Convierte predicciones y valores reales en arrays
        y_true = np.array(soh_real)  # Etiquetas reales
        y_pred = np.array(soh_pred)  # Predicciones del modelo

        # Verifica si hay NaN en los datos
        print(f"NaN en y_true: {np.isnan(y_true).sum()}")
        print(f"NaN en y_pred: {np.isnan(y_pred).sum()}")

        # Verifica si las predicciones son todas iguales
        print(f"Varianza de y_pred: {np.var(y_pred)}")
        print(f"Valores únicos en y_pred: {np.unique(y_pred)}")

        # Si todo está bien, intenta calcular la correlación manualmente
        if np.var(y_pred) > 0 and np.var(y_true) > 0:
            y_pred = y_pred[:len(y_true)]
            pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
            print(f"Correlación Pearson (recalculada): {pearson_corr}")

        return soh_pred

    def plot_soh(self, soh_real, soh_pred):
        """Genera un gráfico comparando SoH real vs. SoH predicho."""
        ciclos = np.arange(len(soh_real))

        plt.figure(figsize=(10, 5))
        plt.plot(ciclos, soh_real, marker='o', linestyle='-', color='blue', label='SoH Real')
        plt.plot(ciclos, soh_pred, marker='s', linestyle='--', color='red', label='SoH Predicho')

        plt.xlabel("Ciclo de carga")
        plt.ylabel("State of Health (SoH)")
        plt.title("Comparación de SoH Real vs. SoH Predicho")
        plt.legend()
        plt.grid(True)
        plt.show()
        r, _ = pearsonr(soh_real, soh_pred)
        print(f"Correlación Pearson: {r:.4f}")

        plt.figure(figsize=(6,6))
        plt.scatter(soh_real, soh_pred, alpha=0.5)
        plt.plot([min(soh_real), max(soh_real)], [min(soh_real), max(soh_real)], color='red', linestyle='--')
        plt.xlabel("SoH Real")
        plt.ylabel("SoH Predicho")
        plt.title("Comparación de SoH Predicho vs. Real")
        plt.grid(True)
        plt.show()

        rmse = mean_squared_error(soh_real, soh_pred) ** 0.5  # sqrt(MSE)
        mae = mean_absolute_error(soh_real, soh_pred)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        plt.show()
        # ##########################################################################



class Inference_SoH_Normal:
    def __init__(self, model_path, input_features, seq_len, n_heads, factor, n_class, n_layers, device="cuda"):
        self.device = device
        self.sand_model = SAnD(input_features, seq_len, n_heads, factor, n_class, n_layers)

        # Cargar los pesos del modelo entrenado
        checkpoint = torch.load(model_path, map_location=device)
        self.sand_model.load_state_dict(checkpoint["model_state_dict"], strict=false)

        self.sand_model.to(device)
        self.sand_model.eval()

    def predict(self, test_loader):
        predictions = []
        soh_real = []
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.clone().detach().to(self.device)
                soh_raw = self.sand_model(x_test)  # Obtener SoH
                soh_pred = soh_raw.cpu().numpy()  # Mover a CPU y convertir a NumPy

                predictions.append(soh_pred)
                soh_real.append(y_test.cpu().numpy())

        # Convertir listas a numpy arrays
        soh_pred = np.concatenate(predictions).flatten()
        soh_real = np.concatenate(soh_real).flatten()

        # Llamar a la función de visualización
        #self.plot_soh(soh_real, soh_pred)

        return soh_pred, soh_real

    def plot_soh(self, soh_real, soh_pred):
        """Genera un gráfico comparando SoH real vs. SoH predicho."""
        ciclos = np.arange(len(soh_real))

        plt.figure(figsize=(10, 5))
        plt.plot(ciclos, soh_real, marker='o', linestyle='-', color='blue', label='SoH Real')
        plt.plot(ciclos, soh_pred, marker='s', linestyle='--', color='red', label='SoH Predicho')

        plt.xlabel("Ciclo de carga")
        plt.ylabel("State of Health (SoH)")
        plt.title("Comparación de SoH Real vs. SoH Predicho")
        plt.legend()
        plt.grid(True)
        plt.show()
        r, _ = pearsonr(soh_real, soh_pred)
        print(f"Correlación Pearson: {r:.4f}")

        plt.figure(figsize=(6,6))
        plt.scatter(soh_real, soh_pred, alpha=0.5)
        plt.plot([min(soh_real), max(soh_real)], [min(soh_real), max(soh_real)], color='red', linestyle='--')
        plt.xlabel("SoH Real")
        plt.ylabel("SoH Predicho")
        plt.title("Comparación de SoH Predicho vs. Real")
        plt.grid(True)
        plt.show()

        rmse = mean_squared_error(soh_real, soh_pred) ** 0.5  # sqrt(MSE)
        mae = mean_absolute_error(soh_real, soh_pred)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        plt.show()


class Inference_SoH_Normal_Improve:
    def __init__(self, model_path, input_features, seq_len, n_heads, factor, n_class, n_layers, device="cuda"):
        self.device = device
        self.sand_model = SAnDImprove(input_features, seq_len, n_heads, factor, n_class, n_layers)

        # Cargar los pesos del modelo entrenado
        checkpoint = torch.load(model_path, map_location=device)
        self.sand_model.load_state_dict(checkpoint["model_state_dict"])

        self.sand_model.to(device)
        self.sand_model.eval()

    def predict(self, test_loader):
        predictions = []
        soh_real = []
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.clone().detach().to(self.device)
                soh_raw = self.sand_model(x_test)  # Obtener SoH
                soh_pred = soh_raw.cpu().numpy()  # Mover a CPU y convertir a NumPy

                predictions.append(soh_pred)
                soh_real.append(y_test.cpu().numpy())

        # Convertir listas a numpy arrays
        soh_pred = np.concatenate(predictions).flatten()
        soh_real = np.concatenate(soh_real).flatten()

        # Llamar a la función de visualización
        #self.plot_soh(soh_real, soh_pred)

        return soh_pred, soh_real


class Inference_SoH_NARX:
    def __init__(self, model_path, input_features, seq_len, n_heads, num_cycles, num_preds, device="cuda"):
        self.device = device
        #self.sand_model = NARX_Transformer_2var_SoloActual(input_features, seq_len, n_heads, num_cycles, num_preds)
        #self.sand_model = NARX_Transformer_2var(input_features, seq_len, n_heads, num_cycles, num_preds)
        self.sand_model = NARX_Transformer(input_features, seq_len, n_heads, num_cycles, num_preds)

        # Cargar los pesos del modelo entrenado
        checkpoint = torch.load(model_path, map_location=device)
        print(checkpoint.keys())

        self.sand_model.load_state_dict(checkpoint["model_state_dict"])

        self.sand_model.to(device)
        self.sand_model.eval()


    def predict(self, test_loader_narx):
        predictions = []
        soh_real = []
        test_total = []
        with torch.no_grad():
            for x_test_narx, cap_test, y_test_narx in test_loader_narx:
                x_test_narx = x_test_narx.to(self.device) if isinstance(x_test_narx, torch.Tensor) else [i_val.to(self.device) for i_val in x_test_narx]
                y_test_narx = y_test_narx.to(self.device)
                b_size = y_test_narx.shape
                test_total += y_test_narx.shape
                cap_test = cap_test.to(self.device)
                soh_pred = self.sand_model(x_test_narx, cap_test)
                predictions.append(soh_pred.cpu())
                soh_real.append(y_test_narx.cpu().numpy())


        # Convertir listas a numpy arrays
        soh_pred = np.concatenate(predictions).flatten()
        soh_real = np.concatenate(soh_real).flatten()

        # Llamar a la función de visualización
        #self.plot_soh(soh_real, soh_pred)

        return soh_pred, soh_real

