from comet_ml import Experiment
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import os
import torch
import pandas as pd
import scipy.io
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import pprint
from SAnD.utils.inference import Inference_SoH_Siamese, Inference_SoH_Normal, Inference_SoH_Normal_Improve, Inference_SoH_NARX
from SAnD.utils.functions import save_example_to_csv, save_example_to_csv_narx, create_cycle_triplets, \
    realizar_inferencia_narx, WrappedModel_NARX,  export_trained_model_to_onnx, ploteo_NARX, \
    ploteo_NN4SOH_aaptado_a_NARX, ploteo_NN4SOH, count_parameters, cargar_modelo_pth_finetuning
import scipy.io as scio
import glob
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib.pyplot as plt
import onnxruntime as ort
import numpy as np
import yaml
from dataset import load_NASA
from SAnD.core.modules import ContrastiveLoss
from SAnD.core.model import SAnD, SAnD_Embedding, SiameseSAnD, SAnDImprove, NARX_Transformer, NARX_Transformer_2var, \
    NARX_Transformer_2var_SoloActual, NARX_Transformer_3var_SoloActual
from SAnD.utils.trainer import NeuralNetworkClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch, gc
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_folder = "dataset/ARC-FY/"  # Modifica esto según tu estructura de carpetas
# mat_files = glob.glob(os.path.join(data_folder, "NMC_CELL_1_processed.mat"))
# # Lista para almacenar los datos concatenados
# raw = []
# # # Cargar cada archivo y agregar sus datos a la lista `raw`
# for mat_file in mat_files:
#     data = scio.loadmat(mat_file)
#     key = list(data.keys())[-1]  # Toma la última clave que suele ser el nombre del dataset
#     extracted_data = data[key][0][0][0][0]  # Extrae los datos
#     raw.extend(extracted_data)  # Concatenar los datos a la lista
#
# print(f"Se han cargado {len(mat_files)} archivos. Tamaño total de raw: {len(raw)}")
#
# pprint.pprint(raw[0])
#
#
# #Creada rama FineTuning para mejorar el entrenamiento con Datasets de químicas similares a las d VE y con ciclos de laboratorio
#
#
#
#
#
#
#
# ########################################################################
# # Fine tuning con ciclos reales de lab
# ########################################################################
#
# csv_paths = [
#     'dataset/Data_finetuning/fila_normalizada_soh_050.csv',
#     'dataset/Data_finetuning/fila_normalizada_soh_073.csv',
#     'dataset/Data_finetuning/fila_normalizada_soh_079.csv'
# ]
#
# samples = []
# labels = []
#
# for path in csv_paths:
#     row = pd.read_csv(path, header=None).values[0]
#
#     V = row[0:400]
#     I = row[400:800]
#     SoH = row[1200]
#
#     # Crear tensor (400, 2) solo con V e I
#     sample = torch.tensor(np.stack((V, I), axis=1), dtype=torch.float32)
#     label = torch.tensor(SoH, dtype=torch.float32)
#
#     samples.append(sample)
#     labels.append(label)
#
# # Convertir a tensores (3, 400, 2) y (3,)
# data = torch.stack(samples)
# targets = torch.tensor(labels)
#
# # Dataset y DataLoader
# dataset = TensorDataset(data, targets)
# dataloader_finetune = DataLoader(dataset, batch_size=1, shuffle=True)


# ########################################################################
# # Fine tuning con Dataset de Laboratorios Sandia
# ########################################################################
# Cargar ambos archivos

# Ruta al fichero combinado
data_path = "dataset/ARC-FY/dataset_final.mat"
# Cargar los datos
data = scipy.io.loadmat(data_path)["data"]  # (1228, 1201)

# Extraer tensores (400, 2) y etiquetas SoH
samples = []
labels = []

for row in data:
    V = row[0:400]
    I = row[400:800]
    SoH = row[800]

    sample = torch.tensor(np.stack((V, I), axis=1), dtype=torch.float32)
    label = torch.tensor(SoH, dtype=torch.float32)

    samples.append(sample)
    labels.append(label)

    plt.figure()
    plt.plot(V, label="Voltaje (V)")
    plt.plot(I, label="Corriente (A)")
    plt.title(f"SoH = {SoH:.3f}")
    plt.xlabel("Tiempo (muestras)")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Crear tensores finales
X_tensor = torch.stack(samples)
y_tensor = torch.tensor(labels)

# Dataset y DataLoader combinados
dataset_finetune = TensorDataset(X_tensor, y_tensor)
dataloader_finetune = DataLoader(dataset_finetune, batch_size=1, shuffle=True)




########################################################################
# Fine-tuning del modelo preentrenado
########################################################################
#
# # Cargar modelo preentrenado
model, checkpoint = cargar_modelo_pth_finetuning(NARX_Transformer_2var_SoloActual,"save_params/trained_model_narx_2var_1ciclo_ok_last.pth")
model.load_state_dict(torch.load('modelo_preentrenado.pth'))
# print(model)

# Congelar todas las capas
for param in model.parameters():
    param.requires_grad = False

# Descongelar solo la cabeza del modelo (ajústalo según tu arquitectura)
for param in model.fc.parameters():
    param.requires_grad = True

# Optimizador con learning rate bajo
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# Función de pérdida
criterion = torch.nn.MSELoss()

# Fine-tuning loop
for epoch in range(50):
    for x, y in dataloader_finetune:
        output = model(x).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.6f}")

########################################################################
# Inferencia con ciclo nuevo: fila_normalizada_soh_079_2.csv
########################################################################

# Cargar nuevo ciclo para inferencia
csv_infer = 'dataset/Data_finetuning/fila_normalizada_soh_050.csv'
row = pd.read_csv(csv_infer, header=None).values[0]

V = row[0:400]
I = row[400:800]
real_SoH = 0.76  # Para comparar si lo deseas

# Preparar input (1, 400, 2)
sample_infer = torch.tensor(np.stack((V, I), axis=1), dtype=torch.float32).unsqueeze(0)

# Poner modelo en modo evaluación
model.eval()
with torch.no_grad():
    predicted_soh = model(sample_infer).item()

print("\nRESULTADO DE INFERENCIA:")
print(f"SoH real:      {real_SoH:.4f}")
print(f"SoH predicho:  {predicted_soh:.4f}")
# Calcular el error absoluto y relativo
error_absoluto = abs(predicted_soh - real_SoH)
error_relativo = (error_absoluto / real_SoH) * 100 if real_SoH != 0 else float('inf')

# Mostrar errores
print(f"\nERROR:")
print(f"Error absoluto: {error_absoluto:.4f}")
print(f"Error relativo: {error_relativo:.2f}%")

# ########################################################################
# #Plotear ciclos de laboratorio *.csv###################################
# ########################################################################
#
# # Cargar el CSV
# # import os
# # print("Directorio actual:", os.getcwd())
# # print("Existe el archivo:", os.path.exists('dataset/Data_finetuning/fila_normalizada_soh_079.csv'))
# csv_path = 'dataset/Data_finetuning/fila_normalizada_soh_050.csv'  # Modifica la ruta si es necesario
# row = pd.read_csv(csv_path, header=None).values[0]
#
# # Separar variables
# V = row[0:400]
# I = row[400:800]
# T = row[800:1200]
# SoH = row[1200]
#
# # Crear tensor con la misma forma (400, 3)
# sample = torch.tensor(np.stack((V, I, T), axis=1), dtype=torch.float32)  # shape: (400, 3)
# label = torch.tensor(SoH, dtype=torch.float32)  # shape: scalar
#
# # Si quieres usarlo junto con el resto de los datos:
# data = torch.stack([sample])  # shape: (1, 400, 3)
# labels = torch.tensor([SoH])  # shape: (1,)
#
#
# # Número de puntos por señal
# n = 400  # Suponiendo que hay n de V, n de I, y 1 SoH
#
# # # Extrae señales
# # V = row[0:400]
# # I = row[400:800]
# # SoH = row[1200]
#
# # Eje temporal ficticio (puedes ajustar si tienes tiempo real)
# x = np.arange(n)
#
# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(x, V, label='Voltaje (V)')
# plt.plot(x, I, label='Intensidad (I)')
# # plt.title(f'Señales de Voltaje e Intensidad - SoH = {SoH:.3f}')
# plt.title(f'SoH_predicho = {predicted_soh:.3f}- SoH_real = {real_SoH:.3f} - Error relativo: {error_relativo:.2f}%')
# plt.xlabel('Muestra')
# plt.ylabel('Valor')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.show()


########################################################################
#CPlotear ciclos de laboratorio *.csv###################################
########################################################################

########################################################################
# Fine tuning con ciclos reales de lab
########################################################################


# def adaptar_a_formato_B0005(cycles_data, soh_labels):
#     """
#     Transforma los ciclos de carga y etiquetas SoH al formato compatible con B0005.mat
#     """
#     data_struct = []
#
#     for i in range(len(cycles_data)):
#         cycle_entry = []
#         cycle_entry.append(['charge'])  # step name
#         cycle_entry.append(np.zeros((1, 1)))  # temp dummy
#         cycle_entry.append(np.zeros((1, 1)))  # temp dummy
#         step_data = np.zeros((7, 1), dtype=object)
#
#         # Cada variable: V, I, T
#         for var_idx in range(3):
#             step_data[var_idx][0] = np.array(cycles_data[i][:, var_idx])
#
#         # tiempo (simulado como intervalo de 10s)
#         step_data[5][0] = np.array([j * 10 for j in range(len(cycles_data[i]))])
#
#         # dummy para temperatura ambiental y otras
#         step_data[3][0] = np.zeros_like(step_data[0][0])
#         step_data[4][0] = np.zeros_like(step_data[0][0])
#         step_data[6][0] = np.ones_like(step_data[0][0]) * soh_labels[i]
#
#         # Añadir estructura final
#         cycle_entry.append([[step_data]])
#         data_struct.append(cycle_entry)
#
#     return np.array([[(data_struct,)]], dtype=object)
#
#
#
# # USO DEL SCRIPT
# # Carga del archivo B0100.mat generado previamente (estructura original)
# mat_original = mat_files
# cycles_data = mat_original['data']
# soh_labels = mat_original['label'].reshape(-1)
#
# # Adaptación
# estructura_B0005 = adaptar_a_formato_B0005(cycles_data, soh_labels)
#
# # Guardar con estructura como la de B0005.mat
# scio.savemat('B0100_compatible.mat', {'B0100': estructura_B0005})
#
# print("Archivo B0100_compatible.mat generado correctamente con la estructura de B0005.")
#






#dataFile = 'dataset/ARC-FY/B0005'   # Modify this path
#raw = scio.loadmat(dataFile)['B0005'][0][0][0][0]

# raw data parsing
cycles = []
labels = []
for i in range(len(raw)):
    if raw[i][0] == ['charge']:
        if i+1 != len(raw) and raw[i+1][0] != ['charge'] and len(raw[i][3][0][0][0][0]) > 850: # discard unfair records
            cycles.append(raw[i][3][0][0])
            if raw[i+1][0] == ['discharge']:
                labels.append(raw[i+1][3][0][0][6][0])
            elif i+2 != len(raw) and raw[i+2][0] == ['discharge']:
                labels.append(raw[i+2][3][0][0][6][0])
cycles.pop()
assert (len(cycles) == len(labels)), 'Number of measurements not matched!'

print(f"cantidad de ciclos: {len(cycles)}")
print(f"cantidad de labels: {len(labels)}")

data = []

# Filtrar solo los ciclos y etiquetas que no estén vacíos
filtered_cycles = []
filtered_labels = []

for i in range(len(labels)):
    if len(labels[i]) > 0:  # Solo conservar si la etiqueta no está vacía
        filtered_cycles.append(cycles[i])
        filtered_labels.append(labels[i])
# Sustituimos las listas originales por las filtradas
cycles = filtered_cycles
labels = filtered_labels

print(f"Nueva cantidad de ciclos: {len(cycles)}")
print(f"Nueva cantidad de labels: {len(labels)}")
for lb in range(len(labels)):
    labels[lb] = labels[lb][0] / 1.856487420818157  # TODO: first (largest) capacity found, but probably not the full cp
labels = labels * 1
for t0 in [0]:
    for cy in cycles:
        t0 = 0
        t = t0
        t_limit = 4000 + t0  # TODO: this parameter can be further tuned
        cursor = 0
        cy_new = []
        while cursor <= len(cy[0][0]) and t <= t_limit:
            while cy[5][0][cursor] <= t:
                cursor += 1
            x1 = cy[5][0][cursor - 1]
            x2 = cy[5][0][cursor]
            point = []
            for i in range(3):
                y1 = cy[i][0][cursor - 1]
                y2 = cy[i][0][cursor]
                y = (t - x1) * (y2 - y1) / (x2 - x1) + y1
                point.append(y)
            cy_new.append(point)
            cursor -= 1
            t += 10
        data.append(cy_new)

print(f"Final Len data: {len(data)}, Len labels: {len(labels)}")

for i in range(len(data)):
    mm = MinMaxScaler()
    data[i] = mm.fit_transform(data[i])
data = np.array(data)
data = data[: , 0:400 ,:]
print(data.shape)
data=torch.from_numpy(data).type(torch.FloatTensor)
labels=torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

data_np = data.numpy() if isinstance(data, torch.Tensor) else data

#################################################### Escalado canal por canal entre -1 y 1
for i in range(3):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(data_np[:, :, i].reshape(-1, 1))
    data_np[:, :, i] = scaled.reshape(data_np.shape[0], data_np.shape[1])

# Volvemos a convertir a tensor
data = torch.from_numpy(data_np).float()
#################################################### Escalado canal por canal entre -1 y 1


##########################################################################################################
#Dataloader para NARX con los datos de la NASA de NARX
##########################################################################################################
# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

# # Access the variables
NUM_CYCLES = cfg['NUM_CYCLES']
NUM_PREDS = cfg['NUM_PREDS']
FEATURE_DIM1 = cfg['FEATURE_DIM1']
FEATURE_DIM2 = cfg['FEATURE_DIM2']
NUM_ATTENTION = cfg['NUM_ATTENTION']
EPOCHS = cfg['EPOCHS']
LEARNING_RATE = cfg['LEARNING_RATE']
BATCH_SIZE = cfg['BATCH_SIZE']

# Load data
train_dataset, test_dataset = load_NASA(folder='NASA_DATA', num_cycles=NUM_CYCLES+NUM_PREDS-1, split_ratio=0.5, scale_data=True)

# Train/test split
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


##########################################################################################################
#Dataloader para NARX con los datos de la NASA de NN4SOH
##########################################################################################################

x_pairs, cap_inputs, y_targets = create_cycle_triplets(data, labels)

######################################################################################
# Mantener solo las dos primeras variables: V (0), I (1) para las olimpIAdas
x_pairs = x_pairs[:, :, :, :2]  # Deja solo V e I, elimina Tª (índice 2)
#######################################################################################
x_train_narx, x_temp_narx, cap_train, cap_temp, y_train_narx, y_temp_narx = train_test_split(
    x_pairs, cap_inputs, y_targets, test_size=0.2, random_state=42, shuffle=False)

x_val_narx, x_test_narx, cap_val, cap_test, y_val_narx, y_test_narx = train_test_split(
    x_temp_narx, cap_temp, y_temp_narx, test_size=0.5, random_state=42, shuffle=False)

train_ds_narx = TensorDataset(x_train_narx, cap_train, y_train_narx)
val_ds_narx = TensorDataset(x_val_narx, cap_val, y_val_narx)
test_ds_narx = TensorDataset(x_test_narx, cap_test, y_test_narx)

train_loader_narx = DataLoader(train_ds_narx, batch_size=32, shuffle=False)
val_loader_narx = DataLoader(val_ds_narx, batch_size=32, shuffle=False)
test_loader_narx = DataLoader(test_ds_narx, batch_size=32, shuffle=False)

##########################################################################################################
#ejemplos de test con el ciclo historico fijo #0
# Seleccionamos el histórico fijo para que sea igual que el current
######################################################################################################

# Creamos nuevos pares con ese histórico fijo combinado con todos los ciclos de test
x_pairs_fixed = []
cap_inputs_fixed = []
y_targets_fixed = []

for i in range(len(x_test_narx)):
    j = random.randint(0, 249)
    current_cycle = x_test_narx[i][1]  # ciclo actual del par
    soh_target = y_test_narx[i]       # SoH objetivo de este ciclo
    historical_cycle = x_test_narx[j][1]  # ciclo actual del par
    historical_soh = cap_test[j]

    x_pair = torch.stack([historical_cycle, current_cycle])  # (2, 400, 3)
    x_pairs_fixed.append(x_pair)
    cap_inputs_fixed.append(historical_soh)  # mismo SoH fijo como entrada
    y_targets_fixed.append(soh_target)

# Convertimos a tensores
x_pairs_fixed_test = torch.stack(x_pairs_fixed)
cap_inputs_fixed_test = torch.stack(cap_inputs_fixed)
y_targets_fixed_test = torch.stack(y_targets_fixed)

# Creamos el nuevo DataLoader con histórico fijo
fixed_test_ds = TensorDataset(x_pairs_fixed_test, cap_inputs_fixed_test, y_targets_fixed_test)
fixed_test_loader = DataLoader(fixed_test_ds, batch_size=32, shuffle=False)

x_pairs_fixed = []
cap_inputs_fixed = []
y_targets_fixed = []

for i in range(len(x_train_narx)-1):
    b = len(x_train_narx)-1
    j = random.randint(0, len(x_train_narx)-1)
    current_cycle = x_train_narx[i+1][1]  # ciclo actual del par
    soh_target = y_train_narx[i+1]       # SoH objetivo de este ciclo
    historical_cycle = x_train_narx[i][1]  # ciclo actual del par
    historical_soh = cap_train[i]

    x_pair = torch.stack([historical_cycle, current_cycle])  # (2, 400, 3)
    x_pairs_fixed.append(x_pair)
    cap_inputs_fixed.append(historical_soh)  # mismo SoH fijo como entrada
    y_targets_fixed.append(soh_target)

# Convertimos a tensores
x_pairs_fixed_train = torch.stack(x_pairs_fixed)
cap_inputs_fixed_train = torch.stack(cap_inputs_fixed)
y_targets_fixed_train = torch.stack(y_targets_fixed)

# Creamos el nuevo DataLoader con histórico fijo
fixed_train_ds = TensorDataset(x_pairs_fixed_train, cap_inputs_fixed_train, y_targets_fixed_train)
fixed_train_loader = DataLoader(fixed_train_ds, batch_size=32, shuffle=False)


x_pairs_fixed = []
cap_inputs_fixed = []
y_targets_fixed = []

for i in range(len(x_val_narx)):
    current_cycle = x_val_narx[i][1]  # ciclo actual del par
    soh_target = y_val_narx[i]       # SoH objetivo de este ciclo
    historical_cycle = x_val_narx[i][1]  # ciclo actual del par
    historical_soh = cap_val[i]


    x_pair = torch.stack([historical_cycle, current_cycle])  # (2, 400, 3)
    x_pairs_fixed.append(x_pair)
    cap_inputs_fixed.append(historical_soh)  # mismo SoH fijo como entrada
    y_targets_fixed.append(soh_target)

# Convertimos a tensores
x_pairs_fixed_val = torch.stack(x_pairs_fixed)
cap_inputs_fixed_val = torch.stack(cap_inputs_fixed)
y_targets_fixed_val = torch.stack(y_targets_fixed)

# Creamos el nuevo DataLoader con histórico fijo
fixed_val_ds = TensorDataset(x_pairs_fixed_val, cap_inputs_fixed_val, y_targets_fixed_val)
fixed_val_loader = DataLoader(fixed_val_ds, batch_size=32, shuffle=False)
##########################################################################################################




##########################################################################################################
#Dataloader para NN4SOH
##########################################################################################################

# Dividir en train, val y test (estratificado si `labels` tiene clases desbalanceadas)
data = data[:, :, :2]  # -> ahora (num_samples, 400, 2)

# Normalización canal a canal (por media y std global)

# # Normaliza canal a canal
means = data.mean(dim=(0, 1), keepdim=True)  # media por canal
stds = data.std(dim=(0, 1), keepdim=True)    # std por canal

data = (data - means) / stds  # normalización global por canal

#Guardar para uso posterior
torch.save({'mean': means, 'std': stds}, 'save_params/normalization_stats.pt')


x_train, x_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

data = data.clone().detach().float()
label = labels.clone().detach().float()
data_ds = TensorDataset(data, label)
data_loader = DataLoader(data_ds, batch_size=32, shuffle=False)

x_train = x_train.clone().detach().float()
x_val = x_val.clone().detach().float()
x_test = x_test.clone().detach().float()

y_train = y_train.clone().detach().float()
y_val = y_val.clone().detach().float()
y_test = y_test.clone().detach().float()

# Shuffle los datos (opcional si `train_test_split` ya los aleatoriza)
indices = torch.randperm(len(x_train))
x_train, y_train = x_train[indices], y_train[indices]

# Crear DataLoaders
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)



##########################################################################
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# =============================
# 1️⃣ Cargar estadísticas de normalización
# =============================
# normalization_stats = torch.load('save_params/normalization_stats.pt')
# means, stds = normalization_stats['mean'], normalization_stats['std']

# =============================
# 2️⃣ Cargar el nuevo ciclo
# =============================
file_path_prueba_real = 'save_params/prueba_real.csv'
prueba_real_data = pd.read_csv(file_path_prueba_real, header=None)

# Separar los datos en voltaje, corriente y SoH
voltage_values_prueba = prueba_real_data.iloc[0, :400].values
current_values_prueba = prueba_real_data.iloc[0, 400:800].values
soh_value_prueba = prueba_real_data.iloc[0, 800]

# =============================
# 3️⃣ Formatear y normalizar el ciclo
# =============================
nuevo_ciclo = np.stack((voltage_values_prueba, current_values_prueba), axis=-1)
nuevo_ciclo_tensor = torch.tensor(nuevo_ciclo, dtype=torch.float32)

# Aplicar normalización
# nuevo_ciclo_tensor = (nuevo_ciclo_tensor - means) / stds
#
# # Expandir dimensión para cumplir con (1, 400, 2)
nuevo_ciclo_tensor = nuevo_ciclo_tensor.unsqueeze(0)

# =============================
# 4️⃣ Añadir al conjunto de test
# =============================
x_test = torch.cat((x_test, nuevo_ciclo_tensor), dim=0)
y_test = torch.cat((y_test, torch.tensor([0.74], dtype=torch.float32)), dim=0)

# =============================
# 5️⃣ Actualizar DataLoader
# =============================
test_ds = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# =============================
# ✅ Verificación final
# =============================
print(f"Dimensiones de x_test: {x_test.shape}")
print(f"Dimensiones de y_test: {y_test.shape}")
print("¡Nuevo ciclo añadido correctamente al DataLoader de test!")



##########################################################################
##########################################################################
# PLoteo de los ciclos
##########################################################################
# ploteo_NARX(val_ds_narx)
#ploteo_NN4SOH_aaptado_a_NARX(train_dataset)
#ploteo_NN4SOH(x_test, y_test)
##########################################################################


in_feature = 3
seq_len = 400
n_heads = 16
factor = 1
num_class = 1
num_layers = 8

hyperparameters = {
    "in_feature": in_feature,
    "seq_len": seq_len,
    "n_heads": n_heads,
    "factor": factor,
    "num_class": num_class,
    "num_layers": num_layers
}

# Load the YAML configuration file
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


clf = NeuralNetworkClassifier(
    SiameseSAnD(SAnD_Embedding(in_feature, seq_len, n_heads, factor, num_class, num_layers)),
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    SAnDImprove(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    NARX_Transformer_2var_SoloActual(feature_dim1,feature_dim2, num_attention, num_cycles, num_preds),
    #NARX_Transformer_3var_SoloActual(feature_dim1,feature_dim2, num_attention, num_cycles, num_preds),
    ContrastiveLoss(),
    nn.MSELoss(),
    nn.MSELoss(),
    #nn.L1Loss(),
    nn.SmoothL1Loss(beta=0.7),  # Cambiar a SmoothL1Loss,
    optim.AdamW,optimizer_config={"lr": 1e-7, "betas": (0.9, 0.98), "eps": 1e-08, "weight_decay": 1e-4},
    # optim.AdamW,optimizer_config={"lr": 1e-6, "betas": (0.9, 0.96), "eps": 1e-08, "weight_decay": 1e-6},
    # optim.SGD, optimizer_config={"lr":1e-6, "momentum": 0.9,"weight_decay": 1e-4},
    #experiment=Experiment("8mKGHiYeg2P7dZEFlvQv3PEzc")
    experiment = Experiment(api_key="Td3ICbNoK8hW14nwxZfp10SGN",
                            project_name="nn4soh",
                            workspace="javiergranadocirce")


)
####################################################
#####cuenta el número de parámetros del modelo
# model = NARX_Transformer_2var(feature_dim1,feature_dim2, num_attention, num_cycles, num_preds)
# count_parameters(model)
#########################################################

inference = True
if inference == True:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    train = False
elif inference == False:
    train = False
    finetuning = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

################################################################################
##### save example to csv for running inference on Raspberry Pi ################
################################################################################
export_csv = False
if export_csv == True:
    inference = False
    train = False
    x_train = x_train.unsqueeze(1)
    x_val = x_val.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    save_example_to_csv_narx(x_test, y_test, 0, filename="save_params/ciclo_de_carga_narx_2var_1ciclo_1_last.csv")
################################################################################



#################################################################################################################3
# Training process
###############################################################################################################

if train == True:

    if finetuning == True:
        modelo, checkpoint = cargar_modelo_pth_finetuning(NARX_Transformer_2var_SoloActual,"save_params/trained_model_narx_2var_1ciclo_ok_last.pth")
        modelo = modelo.to(device)
        epoch = checkpoint['epoch']
        optimizer = torch.optim.Adam(modelo.parameters())  # Usando el lr guardado
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        clf.fit_NARX_Transformer_finetuning(epoch, optimizer, modelo, x_train, y_train, x_val, y_val, x_test, y_test,
                                 {"train_narx": train_loader,
                                  "val_narx": val_loader,
                                  "test_narx": test_loader},
                                 epochs=2
                                 )
        # ##################################################################################################
        # # # save pth model when finetuning process is completed
        # #############################################################################################
        clf.save_to_file_Narx_finetuning("trained_model_narx_2var_1ciclo_finetuning.pth")
        #############################################################################################

    elif finetuning == False:
        # # training network Normal
        # clf.fit_normal(x_train, y_train, x_val, y_val, x_test, y_test,
        #         {"train": train_loader,
        #       "val": val_loader,
        #       "test": test_loader},
        #       epochs=80
        # )

        # # #training network Improve
        # clf.fit_normal_improve(x_train, y_train, x_val, y_val, x_test, y_test,
        #          {"train": train_loader,
        #       "val": val_loader,
        #       "test": test_loader},
        #       epochs=80
        # )
        # # #
        # training network Siamese
        # clf.fit_siamese(x_train, y_train, x_val, y_val, x_test, y_test,
        #             {"train": train_loader,
        #         "val": val_loader,
        #         "test": test_loader},
        #         epochs=80
        # )

        # # training network NARX
        x_train = x_train.unsqueeze(1)
        x_val = x_val.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        #
        clf.fit_NARX_Transformer(x_train, y_train, x_val, y_val, x_test, y_test,
                    {"train_narx": train_loader,
                "val_narx": val_loader,
                "test_narx": test_loader},
                epochs=1000
        )


        # clf.fit_NARX_Transformer3V(x_train, y_train, x_val, y_val, x_test, y_test,
        #              {"train_narx": train_loader,
        #               "val_narx": val_loader,
        #               "test_narx": test_loader},
        #              epochs=200
        # )



    # ##################################################################################################
    # # # save pth model when training process is completed
    # #############################################################################################
        clf.save_to_file_Narx("save_params/")
    #############################################################################################

    # ##################################################################################################
    # # # Export trained model to onnx after save pth model
    # #############################################################################################
        export_trained_model_to_onnx(feature_dim1,feature_dim2, num_attention, num_cycles, num_preds)
    #############################################################################################


# ##################################################################################################
# # #Inferencia en PC
# #############################################################################################

# 🔹 Ejemplo de uso
if inference ==  True:

    modo = "pth"  # Cambia a "pth" para usar el modelo original
    modelo ="save_params/trained_model_narx_2var_1ciclo_ok_last.pth"
    test_ds = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    realizar_inferencia_narx(test_loader, x_test, y_test, modo, modelo)
    #realizar_inferencia_narx(test_loader_narx, x_test_narx, cap_test, y_test_narx, modo, modelo)
    #realizar_inferencia_narx(fixed_test_loader,x_pairs_fixed_test, cap_inputs_fixed_test, y_targets_fixed_test, modo, modelo)



