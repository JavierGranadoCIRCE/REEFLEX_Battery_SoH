import torch
import numpy as np
import random
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from SAnD.core.model import NARX_Transformer_2var, NARX_Transformer, NARX_Transformer_2var_SoloActual
from SAnD.utils.inference import Inference_SoH_NARX

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

# # Access the variables
num_cycles = cfg['NUM_CYCLES']
num_preds = cfg['NUM_PREDS']
feature_dim1 = cfg['FEATURE_DIM1']
feature_dim2 = cfg['FEATURE_DIM2']
num_attention = cfg['NUM_ATTENTION']
EPOCHS = cfg['EPOCHS']
LEARNING_RATE = cfg['LEARNING_RATE']
BATCH_SIZE = cfg['BATCH_SIZE']


def positional_encoding(n_positions: int, hidden_dim: int) -> torch.Tensor:
    def calc_angles(pos, i):
        rates = 1 / np.power(10000, (2*(i // 2)) / np.float32(hidden_dim))
        return pos * rates

    rads = calc_angles(np.arange(n_positions)[:, np.newaxis], np.arange(hidden_dim)[np.newaxis, :])

    rads[:, 0::2] = np.sin(rads[:, 0::2])
    rads[:, 1::2] = np.cos(rads[:, 1::2])

    pos_enc = rads[np.newaxis, ...]
    pos_enc = torch.tensor(pos_enc, dtype=torch.float32, requires_grad=False)
    return pos_enc


def dense_interpolation(batch_size: int, seq_len: int, factor: int) -> torch.Tensor:
    W = np.zeros((factor, seq_len), dtype=np.float32)
    for t in range(seq_len):
        s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
        for m in range(factor):
            tmp = np.array(1 - (np.abs(s - (1 + m)) / factor), dtype=np.float32)
            w = np.power(tmp, 2, dtype=np.float32)
            W[m, t] = w

    W = torch.tensor(W, requires_grad=False).float().unsqueeze(0)
    return W.repeat(batch_size, 1, 1)


def subsequent_mask(size: int) -> torch.Tensor:
    """
    from Harvard NLP
    The Annotated Transformer

    http://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking

    :param size: int
    :return: torch.Tensor
    """
    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("float32")
    mask = torch.from_numpy(mask) == 0
    return mask.float()

def generar_pares_aleatorios(x_train, y_train, umbral_soh=0.02):
    """
    Genera pares aleatorios de ejemplos de carga junto con sus etiquetas 0 o 1
    según si tienen un estado de salud similar o diferente.

    Parámetros:
    - data: lista de ciclos de carga (cada elemento es una secuencia de carga)
    - labels: lista de valores de SoH correspondientes a cada ciclo
    - num_pares: número total de pares a generar
    - umbral_soh: diferencia máxima entre SoH para considerar que es el mismo estado

    Retorna:
    - X_pairs: lista con los pares de ciclos de carga
    - y_pairs: lista con etiquetas 0 o 1 según su estado de salud
    """

    X_pairs = []
    y_pairs = []

    total_ciclos = len(x_train)


    # Seleccionar dos ciclos aleatorios
    i, j = random.sample(range(total_ciclos), 2)

    ciclo_1 = x_train[i]
    ciclo_2 = x_train[j]
    soh_1 = y_train[i]
    soh_2 = y_train[j]

    # Asignar etiqueta: 1 si los SoH son similares, 0 si son diferentes
    y = 1 if abs(soh_1 - soh_2) < umbral_soh else 0

    # Guardar el par y su etiqueta
    X_pairs.append((ciclo_1, ciclo_2))
    y_pairs.append(y)
    x1 = ciclo_1
    x2 = ciclo_2
    y_cont = y
    if x1.dim() == 2:  # Si x1 tiene la forma (400, 3)
        x1 = x1.unsqueeze(0)  # Convierte en (1, 400, 3)
    if x2.dim() == 2:  # Si x2 tiene la forma (400, 3)
        x2 = x2.unsqueeze(0)  # Convierte en (1, 400, 3)
    return x1, x2, y_cont


def create_cycle_triplets(data, labels):
    x_pairs = []
    capacities = []
    y_targets = []

    for i in range(1, len(data) - 1):
        x_pair = torch.stack([data[i], data[i+1]])  # (2, 400, 3)
        x_pairs.append(x_pair)
        capacities.append(labels[i])       # SoH del ciclo anterior
        y_targets.append(labels[i+1])      # SoH del ciclo actual (target)

    x_pairs = torch.stack(x_pairs)
    capacities = torch.tensor(capacities, dtype=torch.float32).unsqueeze(1)  # (N-2, 1)
    y_targets = torch.tensor(y_targets, dtype=torch.float32)
    return x_pairs, capacities, y_targets




def save_example_to_csv(x_train, y_train, example_idx, filename="ciclo_de_carga.csv"):
    """
    Guarda un ejemplo de x_train con su correspondiente etiqueta de y_train en un archivo CSV.

    Parámetros:
    - x_train: Tensor de entrada con forma (N, 400, 3).
    - y_train: Tensor de etiquetas con forma (N,).
    - example_idx: Índice del ejemplo a guardar.
    - filename: Nombre del archivo CSV de salida (por defecto "example_data_with_label.csv").
    """

    # Verificar que el índice es válido
    if example_idx < 0 or example_idx >= len(x_train):
        raise ValueError(f"Índice fuera de rango: {example_idx}. Debe estar entre 0 y {len(x_train) - 1}.")

    # Aplanar el tensor del ejemplo para convertirlo en un vector 1D de longitud 1200
    example_data = x_train[example_idx].reshape(-1)  # De [400, 3] a [1200]

    # Tomar la etiqueta correspondiente
    example_label = y_train[example_idx]

    # Combinar los datos de entrada con la etiqueta
    data_with_label = np.append(example_data, example_label)  # Unir características y etiqueta

    # Convertir a un DataFrame de pandas
    df = pd.DataFrame(data_with_label.reshape(1, -1))

    # Guardar en un archivo CSV sin encabezados ni índice
    df.to_csv(filename, header=False, index=False)

    print(f"Ejemplo {example_idx} guardado en {filename}")



def save_example_to_csv_narx(x_pair, y_target, example_idx, filename="ciclo_de_carga.csv"):
    """
    Guarda un ejemplo de entrada (x_pair) y su etiqueta (y_target) en un CSV.

    - x_pair: Tensor con forma (N, 400, 2)
    - y_target: Tensor con forma (N,)
    - example_idx: Índice del ejemplo a guardar
    - filename: Nombre del archivo CSV de salida
    """

    if example_idx < 0 or example_idx >= len(x_pair):
        raise ValueError(f"Índice fuera de rango: {example_idx}")

    # x_pair: (400, 2) → flatten → 800
    example_data = x_pair[example_idx].reshape(-1).numpy()

    # y_target: (1,) → float
    label = y_target[example_idx].numpy()

    # Concatenar todo: [x_pair_flattened, label]
    all_data = np.concatenate([example_data, [label]])

    # Guardar en un DataFrame para CSV
    df = pd.DataFrame(all_data.reshape(1, -1))
    df.to_csv(filename, header=False, index=False)

    print(f"Ejemplo {example_idx} guardado en {filename}")

def realizar_inferencia_narx(loader, x_test, y_test,  modo="onnx", modelo=None):
    """Realiza la inferencia usando ONNX o PyTorch y calcula métricas."""
    mae_total,  mse_sum = 0, 0
    smape_total, mape_total = 0, 0
    real_values = []
    pred_values = []
    if modo == "onnx":
        pred_values = []
        real_values = []
        mae_total, mse_total, mape_total, smape_total = 0, 0, 0, 0
        session, input_names = cargar_modelo(modo, modelo)
        for i in range(len(x_test)):
            x_sample = x_test[i].numpy().astype(np.float32)[np.newaxis, ...]       # (1, 2, 400, 3)
            #cap_sample = cap_input_test[i].numpy().astype(np.float32)[np.newaxis]  # (1, 1)

            # Inferencia con ONNX
            output = session.run(None, {
                input_names[0]: x_sample,
                #input_names[1]: cap_sample
            })[0]
            pred = output[0][0]  # (1,)
            real = y_test[i].item()

            # Guardar predicción y etiqueta real
            pred_values.append(pred)
            real_values.append(real)

            # Cálculo de errores
            mse_sum += (real - pred) ** 2
            mae_total += np.abs(pred - real)
            mse_total += (pred - real) ** 2

            if real != 0:
                mape_total += np.abs((pred - real) / real)
            smape_total += np.abs(pred - real) / ((np.abs(pred) + np.abs(real)) / 2)
            # Mostrar resultado parcial
            print(f"Ejemplo {i + 1}/{len(x_test)} -> Predicción: {pred:.4f}, Real: {real:.4f}")

    if modo == "pth":
        predicciones = []
        etiquetas_reales = []
        mae_total, mse_total, mape_total, smape_total = 0, 0, 0, 0
        inference_model = Inference_SoH_NARX(modelo, input_features=feature_dim1, seq_len=feature_dim2, n_heads=num_attention, num_cycles = num_cycles, num_preds=num_preds)
        soh_predictions = inference_model.predict(loader)
        #Inference SoH ###############################
        # real = soh_predictions[1]
        real_values = []
        pred_values = []

        for idx in range(len(x_test)):
            pred = soh_predictions[0][idx]
            real = soh_predictions[1][idx]

            # Guardar valores para graficar
            real_values.append(real)
            pred_values.append(pred)

            # Cálculo de errores
            mae_total += np.abs(real - pred)
            mse_sum += (real - pred) ** 2
            if real != 0:
                mape_total += np.abs((pred - real) / real)
            smap_sup = pred - real
            smap_inf = (np.abs(pred) + np.abs(real)) / 2
            smape_total += np.abs(smap_sup / smap_inf)

            # Mostrar resultado parcial
            print(f"Ejemplo {idx + 1}/{len(x_test)} -> Predicción: {pred}, Etiqueta Real: {real}")

    # Graficar los valores reales y predichos
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(real_values[:])), real_values[:], label="Real", color="blue", marker="o")
    plt.scatter(range(len(pred_values[:])), pred_values[:], label="Predicho", color="red", marker="x")


    # Etiquetas y título
    plt.xlabel("Índice de muestra")
    plt.ylabel("State of Health (SoH)")
    plt.title("Comparación de SoH Real vs Predicho")
    plt.legend()
    plt.show()
    # Cálculo de métricas
    mae = mae_total / len(x_test)
    mse = mse_sum / len(x_test)
    rmse = np.sqrt(mse)
    smape = smape_total / len(x_test)
    # Calcula el MAPE promedio
    mape = mape_total / len(x_test)
    #  Multiplica por 100 para tener el resultado en porcentaje
    # mape_total*= 100
    #mape = (mape_total / len(x_test)) * 100

    print("\nMétricas finales:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"ERROR TOTAL: {smape * 100:.1f}%")


def cargar_modelo_pth_finetuning(model_class, path_modelo):
    modelo = model_class(feature_dim1, feature_dim2, num_attention, num_cycles, num_preds)
    checkpoint = torch.load(path_modelo, map_location=torch.device('cuda'))
    print(checkpoint.keys())  # Verifica las claves del checkpoint
    modelo.load_state_dict(checkpoint['model_state_dict'])  # Cargar solo el modelo
    return modelo, checkpoint




def cargar_modelo(modo="onnx", modelo = None):
    """Carga el modelo según el modo especificado."""
    if modo == "onnx":
        # session = ort.InferenceSession("save_params/trained_model_normal.onnx")
        session = ort.InferenceSession(modelo)
        input_names = [inp.name for inp in session.get_inputs()]
        print(f"[INFO] Entradas del modelo ONNX: {input_names}")
        return session, input_names
    else:
        raise ValueError("Modo no reconocido. Usa 'onnx' o 'pth'.")

def export_trained_model_to_onnx(feature_dim1,feature_dim2, num_attention, num_cycles, num_preds):
    modelo = NARX_Transformer_2var_SoloActual(feature_dim1,feature_dim2, num_attention, num_cycles, num_preds)
    checkpoint = torch.load("save_params/trained_model_narx_2var_1ciclo_ok_last.pth", map_location="cpu")
    modelo.load_state_dict(checkpoint["model_state_dict"], strict=False)
    modelo.eval()
    wrapped_model = WrappedModel_NARX(modelo)  # Envolver modelo con sigmoide
    # Dummy inputs (para NARX)
    dummy_x_test = torch.randn(1, 400, 2)


    torch.onnx.export(
        wrapped_model,
        (dummy_x_test),  # ahora son dos entradas
        "save_params/trained_model_narx_2var_1ciclo_ok_last.onnx",
        input_names=["x_pair"],
        output_names=["soh_pred"],
        opset_version=17,
        dynamic_axes={
            "x_pair": {0: "batch_size"},
            "soh_pred": {0: "batch_size"}
        }
    )
    print(f"Ejemplo onnx guardado")

class WrappedModel_NARX(nn.Module):
    def __init__(self, base_model):
        super(WrappedModel_NARX, self).__init__()
        self.base_model = base_model

    def forward(self, x_test):
        return self.base_model(x_test)




##########################################################################
# PLoteo de los ciclos de carga del dataset completo de NARX
def ploteo_NARX(ciclos):
    variables = ["Tensión (V)", "Corriente (A)", "Temperatura (°C)"]
    colores = ["b", "r", "g"]  # Azul, rojo y verde

    # Recorrer todos los ejemplos del dataset
    for sample_idx in range(len(ciclos)):
        x_train, cap_inputs_fixed, y_train = ciclos[sample_idx]  # x_train: (num_cycles, 400, 3), y_train: (num_cycles,)

        # Recorrer los ciclos de carga dentro de este ejemplo
        for i in range(x_train.shape[0]):
            plt.figure(figsize=(10, 5))
            for j in range(3):
                plt.plot(x_train[i, :, j], color=colores[j], label=variables[j])

            soh_value = y_train.item()
            plt.xlabel("Tiempo (puntos de muestreo)")
            plt.ylabel("Valor")
            plt.title(f"Ejemplo {sample_idx+1}, Ciclo {i+1} - SoH: {soh_value:.2f}%")
            plt.legend()
            plt.grid()
            plt.show()
            input("Presiona Enter para ver el siguiente ciclo...")
            plt.close()
##########################################################################



##########################################################################
# PLoteo de los ciclos de carga del dataset completo de NN4SOH adaptado a NARX
def ploteo_NN4SOH_aaptado_a_NARX(ciclos):
    variables = ["Tensión (V)", "Corriente (A)", "Temperatura (°C)"]
    colores = ["b", "r", "g"]  # Azul, rojo y verde

    # Recorrer todos los ejemplos del dataset
    for sample_idx in range(len(ciclos)):
        x_val, y_val = ciclos[sample_idx]  # x_train: (num_cycles, 400, 3), y_train: (num_cycles,)

        # Recorrer los ciclos de carga dentro de este ejemplo
        for i in range(x_val.shape[0]):
            plt.figure(figsize=(10, 5))
            for j in range(3):
                plt.plot(x_val[i, :, j], color=colores[j], label=variables[j])

            soh_value = y_val[i]
            plt.xlabel("Tiempo (puntos de muestreo)")
            plt.ylabel("Valor")
            plt.title(f"Ejemplo {sample_idx+1}, Ciclo {i+1} - SoH: {soh_value:.2f}%")
            plt.legend()
            plt.grid()
            plt.show()
            input("Presiona Enter para ver el siguiente ciclo...")
            plt.close()
##########################################################################



# ##########################################################################
# # PLoteo de los ciclos de carga del dataset completo de NN4SOH
#
#
# # Etiquetas de las variables
def ploteo_NN4SOH(ciclos_x, ciclos_y):
    variables = ["Tensión (V)", "Corriente (A)"]
    colores = ["b", "r"]  # Azul, rojo y verde

    for i in range(ciclos_x.shape[0]):  # Recorremos los ciclos de carga
        plt.figure(figsize=(10, 5))

        # Dibujar las 2 variables en distintos colores
        for j in range(2):
            plt.plot(ciclos_x[i, :, j], color=colores[j], label=variables[j])

        soh_value = ciclos_y[i]  # Obtener el SoH del ciclo actual
        plt.xlabel("Tiempo (puntos de muestreo)")
        plt.ylabel("Valor")
        plt.title(f"Ciclo de carga {i+1} - SoH: {soh_value:.2f}%")  # Agregar el SoH en el título
        plt.legend()
        plt.grid()


        plt.show()

        input("Presiona Enter para ver el siguiente ciclo...")  # Espera antes de mostrar el siguiente gráfico
        plt.close()
# # PLoteo de los coclos de carga del dataset completo
# ##########################################################################


##########calculo parámetros del modelo###############
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parámetros: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
##########calculo parámetros del modelo###############


class ScheduledOptimizer:
    """
    Reference: `jadore801120/attention-is-all-you-need-pytorch \
    <https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py>`_
    """
    def __init__(self, optimizer, d_model: int, warm_up: int) -> None:
        self._optimizer = optimizer
        self.warm_up = warm_up
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> np.array:
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.warm_up, -1.5) * self.n_current_steps
        ])

    def get_lr(self):
        lr = self.init_lr * self._get_lr_scale()
        return lr

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.get_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return self._optimizer.state_dict()
