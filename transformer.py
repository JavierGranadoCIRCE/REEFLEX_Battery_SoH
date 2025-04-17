from comet_ml import Experiment
import torch.nn as nn
import torch.optim as optim
from SAnD.utils.inference import Inference_SoH_Siamese, Inference_SoH_Normal, Inference_SoH_Normal_Improve, Inference_SoH_NARX
from SAnD.utils.functions import save_example_to_csv, save_example_to_csv_narx, create_cycle_triplets
import scipy.io as scio
import glob
import os
import matplotlib.pyplot as plt
import onnxruntime as ort
import numpy as np
import yaml
from dataset import load_NASA
from SAnD.core.modules import ContrastiveLoss
from SAnD.core.model import SAnD, SAnD_Embedding, SiameseSAnD, SAnDImprove, NARX_Transformer
from SAnD.utils.trainer import NeuralNetworkClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch, gc
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = "dataset/ARC-FY/"  # Modifica esto según tu estructura de carpetas
mat_files = glob.glob(os.path.join(data_folder, "*.mat"))
# Lista para almacenar los datos concatenados
raw = []
# Cargar cada archivo y agregar sus datos a la lista `raw`
for mat_file in mat_files:
    data = scio.loadmat(mat_file)
    key = list(data.keys())[-1]  # Toma la última clave que suele ser el nombre del dataset
    extracted_data = data[key][0][0][0][0]  # Extrae los datos
    raw.extend(extracted_data)  # Concatenar los datos a la lista

print(f"Se han cargado {len(mat_files)} archivos. Tamaño total de raw: {len(raw)}")

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
#x_pairs = x_pairs[:, :, :, :2]  # Deja solo V e I, elimina Tª (índice 2)
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
    current_cycle = x_test_narx[i][1]  # ciclo actual del par
    soh_target = y_test_narx[i]       # SoH objetivo de este ciclo
    historical_cycle = x_test_narx[i][1]  # ciclo actual del par
    historical_soh = y_test_narx[i].unsqueeze(-1)

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

for i in range(len(x_train_narx)):
    current_cycle = x_train_narx[i][1]  # ciclo actual del par
    soh_target = y_train_narx[i]       # SoH objetivo de este ciclo
    historical_cycle = x_train_narx[i][1]  # ciclo actual del par
    historical_soh = y_train_narx[i].unsqueeze(-1)

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
    historical_soh = y_val_narx[i].unsqueeze(-1)


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
x_train, x_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

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


# ##########################################################################
# # PLoteo de los ciclos de carga del dataset completo de NARX
#
# variables = ["Tensión (V)", "Corriente (A)", "Temperatura (°C)"]
# colores = ["b", "r", "g"]  # Azul, rojo y verde
#
# # Recorrer todos los ejemplos del dataset
# for sample_idx in range(len(val_ds_narx)):
#     x_train, cap_inputs_fixed, y_train = val_ds_narx[sample_idx]  # x_train: (num_cycles, 400, 3), y_train: (num_cycles,)
#
#     # Recorrer los ciclos de carga dentro de este ejemplo
#     for i in range(x_train.shape[0]):
#         plt.figure(figsize=(10, 5))
#         for j in range(3):
#             plt.plot(x_train[i, :, j], color=colores[j], label=variables[j])
#
#         soh_value = y_train.item()
#         plt.xlabel("Tiempo (puntos de muestreo)")
#         plt.ylabel("Valor")
#         plt.title(f"Ejemplo {sample_idx+1}, Ciclo {i+1} - SoH: {soh_value:.2f}%")
#         plt.legend()
#         plt.grid()
#         plt.show()
#         input("Presiona Enter para ver el siguiente ciclo...")
#         plt.close()
# ##########################################################################



# ##########################################################################
# # PLoteo de los ciclos de carga del dataset completo de NN4SOH adaptado a NARX
#
# variables = ["Tensión (V)", "Corriente (A)", "Temperatura (°C)"]
# colores = ["b", "r", "g"]  # Azul, rojo y verde
#
# # Recorrer todos los ejemplos del dataset
# for sample_idx in range(len(train_dataset)):
#     x_train, y_train = train_dataset[sample_idx]  # x_train: (num_cycles, 400, 3), y_train: (num_cycles,)
#
#     # Recorrer los ciclos de carga dentro de este ejemplo
#     for i in range(x_train.shape[0]):
#         plt.figure(figsize=(10, 5))
#         for j in range(3):
#             plt.plot(x_train[i, :, j], color=colores[j], label=variables[j])
#
#         soh_value = y_train[i]
#         plt.xlabel("Tiempo (puntos de muestreo)")
#         plt.ylabel("Valor")
#         plt.title(f"Ejemplo {sample_idx+1}, Ciclo {i+1} - SoH: {soh_value:.2f}%")
#         plt.legend()
#         plt.grid()
#         plt.show()
#         input("Presiona Enter para ver el siguiente ciclo...")
#         plt.close()
# ##########################################################################



# # ##########################################################################
# # # PLoteo de los ciclos de carga del dataset completo de NN4SOH
# #
# #
# # # Etiquetas de las variables
# variables = ["Tensión (V)", "Corriente (A)", "Temperatura (°C)"]
# colores = ["b", "r", "g"]  # Azul, rojo y verde
#
# for i in range(x_train.shape[0]):  # Recorremos los ciclos de carga
#     plt.figure(figsize=(10, 5))
#
#     # Dibujar las 3 variables en distintos colores
#     for j in range(3):
#         plt.plot(x_train[i, :, j], color=colores[j], label=variables[j])
#
#     soh_value = y_train[i]  # Obtener el SoH del ciclo actual
#     plt.xlabel("Tiempo (puntos de muestreo)")
#     plt.ylabel("Valor")
#     plt.title(f"Ciclo de carga {i+1} - SoH: {soh_value:.2f}%")  # Agregar el SoH en el título
#     plt.legend()
#     plt.grid()
#
#
#     plt.show()
#
#     input("Presiona Enter para ver el siguiente ciclo...")  # Espera antes de mostrar el siguiente gráfico
#     plt.close()
# # # PLoteo de los coclos de carga del dataset completo
# # ##########################################################################




#################################################################################################################3
# Training
###############################################################################################################
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
    NARX_Transformer(feature_dim1,feature_dim2, num_attention, num_cycles, num_preds),
    ContrastiveLoss(),
    nn.MSELoss(),
    nn.MSELoss(),
    nn.L1Loss(),
    #nn.SmoothL1Loss(beta=0.1),  # Cambiar a SmoothL1Loss,
    optim.AdamW,optimizer_config={"lr": 1e-7, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
    # optim.AdamW,optimizer_config={"lr": 1e-6, "betas": (0.9, 0.96), "eps": 1e-08, "weight_decay": 1e-6},
    # optim.SGD, optimizer_config={"lr":1e-6, "momentum": 0.9,"weight_decay": 1e-4},
    #experiment=Experiment("8mKGHiYeg2P7dZEFlvQv3PEzc")
    experiment = Experiment(api_key="Td3ICbNoK8hW14nwxZfp10SGN",
                            project_name="nn4soh",
                            workspace="javiergranadocirce")


)


##########calculo parámetros del modelo###############
# def count_parameters(model):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total, trainable
#
# model = NARX_Transformer(16, 16, 16, 2, 1)
# total_params, trainable_params = count_parameters(model)
# print(f"Total de parámetros: {total_params:,}")
# print(f"Parámetros entrenables: {trainable_params:,}")
##########calculo parámetros del modelo###############


inference = True
if inference == True:
    train = False
elif inference == False:
    train = True
    torch.cuda.empty_cache()
    gc.collect()

export_csv = False
if export_csv == True:
    inference = False
    train = False


    #####save example to csv####################################################
    #save_example_to_csv(x_test, y_test, 2490, filename="save_params/ciclo_de_carga.csv")
    save_example_to_csv_narx(x_test_narx, cap_test, y_test_narx, 8, filename="save_params/ciclo_de_carga_narx_8.csv")
    #####save example to csv####################################################


if train == True:
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

    # training network NARX
    clf.fit_NARX_Transformer(x_pairs_fixed_train, y_targets_fixed_train, x_pairs_fixed_val, y_targets_fixed_val, x_pairs_fixed_test, y_targets_fixed_test,
                {"train_narx": fixed_train_loader,
            "val_narx": fixed_val_loader,
            "test_narx": fixed_test_loader},
            epochs=2000
    )



    # # # #
    # # # #
    # # # #
    # # # # #Inference SoH Siames ###############################
    # # # # #inference_model = Inference_SoH_Siamese("save_params/trained_model_siamese.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
    # # # # #soh_predictions = inference_model.predict(test_loader)
    # # # # #Inference SoH ###############################
    # # # #
    # # # # #Inference SoH Normal ###############################
    # # # # # inference_model = Inference_SoH_Normal("save_params/trained_model_normal.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
    # # # # # soh_predictions = inference_model.predict(test_loader)
    # # # # #Inference SoH ###############################
    # # # #
    # # # # # evaluating
    # # # # # clf.restore_from_file("save_params/trained model.pth", "cuda")
    # # # # # clf.evaluate(test_loader)
    # # # #
    # # # # save
    #clf.save_to_file_normal("save_params/")
    #clf.save_to_file_normal_improve("save_params/")
    #clf.save_to_file_siamese("save_params/")
    clf.save_to_file_Narx("save_params/")
    # #
    # # #
    # # #
    # # #
    # # # # Conversión a ONNX
    # # # # Cargar el modelo entrenado



    class WrappedModel(nn.Module):
        def __init__(self, model):
            super(WrappedModel, self).__init__()
            self.model = model
            self.sigmoid = nn.Sigmoid()  # Agregar sigmoide

        def forward(self, x):
            return self.sigmoid(self.model(x))  # Aplicar sigmoide después del modelo

    class WrappedModel_NARX(nn.Module):
        def __init__(self, base_model):
            super(WrappedModel_NARX, self).__init__()
            self.base_model = base_model

        def forward(self, x_pair, cap_input):
            return self.base_model(x_pair, cap_input)


    #modelo = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers)
    #modelo = SAnDImprove(in_feature, seq_len, n_heads, factor, num_class, num_layers)
    modelo = NARX_Transformer(feature_dim1,feature_dim2, num_attention, num_cycles, num_preds)
    #modelo = SAnD_Embedding(in_feature, seq_len, n_heads, factor, num_class, num_layers)
    # # # # Verificar los atributos de modelo_siamese
    # print(modelo_siamese)
    # # # # # Verificar los atributos de modelo_normal
    # print(modelo)
    # # # # # Copiar pesos de la parte compartida del modelo siamesa al modelo normal
    # # # # Transferir pesos del modelo siamesa al modelo normal
    # modelo.encoder.load_state_dict(modelo_siamese.sand.encoder.state_dict(), strict=False)  # Transferir encoder
    # modelo.dense_interpolation.load_state_dict(modelo_siamese.sand.dense_interpolation.state_dict(), strict=False)  # Transferir dense_interpolation
    # # #
    # # #
    # # # print("Pesos transferidos correctamente.")
    # # # print(modelo)
    # # #
    # # # # 2. Cargar el diccionario de estado correctamente

    # # checkpoint = torch.load("save_params/trained_model_normal_old.pth", map_location="cpu")
    # # print(checkpoint.keys())  # Ver qué hay dentro
    # # if "hyperparameters" in checkpoint:  # Si guardaste los hiperparámetros
    # #     print(checkpoint["hyperparameters"])
    #checkpoint = torch.load("save_params/trained_model_normal_improve.pth", map_location="cpu")
    #checkpoint = torch.load("save_params/trained_model_normal.pth", map_location="cpu")
    checkpoint = torch.load("save_params/trained_model_narx.pth", map_location="cpu")
    modelo.load_state_dict(checkpoint["model_state_dict"], strict=False)
    modelo.eval()
    #wrapped_model = WrappedModel(modelo)  # Envolver modelo con sigmoide
    wrapped_model = WrappedModel_NARX(modelo)  # Envolver modelo con sigmoide
    # # #
    # # # # # Crear un dummy input (ajusta el tamaño según tu entrada real)
    # # # #
    #input_shape = (400, 3)
    #dummy_input = torch.randn(1, *input_shape)

    # Dummy inputs (para NARX)
    dummy_x_pair = torch.randn(1, 2, 400, 3)
    dummy_cap_input = torch.randn(1, 1)

    # # # #
    # # # # # Exportar a ONNX
    #torch.onnx.export(modelo, dummy_input, "save_params/trained_model_normal.onnx", opset_version=13)
    #torch.onnx.export(wrapped_model, dummy_input, "save_params/trained_model_normal_improve.onnx", opset_version=13)
    # # torch.onnx.export(modelo, dummy_input, "save_params/trained_model_normal_improve_old.onnx", opset_version=13)

    torch.onnx.export(
        wrapped_model,
        (dummy_x_pair, dummy_cap_input),  # ahora son dos entradas
        "save_params/trained_model_narx.onnx",
        input_names=["x_pair", "cap_input"],
        output_names=["soh_pred"],
        opset_version=17,
        dynamic_axes={
            "x_pair": {0: "batch_size"},
            "cap_input": {0: "batch_size"},
            "soh_pred": {0: "batch_size"}
        }
    )



    # # #
    # # # # #
    # #
    # # #
# ##################################################################################################
# # #Inferencia en PC
# #############################################################################################
#
def cargar_modelo(modo="onnx", modelo = None):
    """Carga el modelo según el modo especificado."""
    if modo == "onnx":
        # session = ort.InferenceSession("save_params/trained_model_normal.onnx")
        session = ort.InferenceSession(modelo)
        input_name = session.get_inputs()[0].name
        return session, input_name
    # elif modo == "pth":
    #     #inference_model = Inference_SoH_Normal_Improve("save_params/trained_model_normal_improve.pth", input_features=3, seq_len=400, n_heads=64, factor=32, n_class=1, n_layers=12)
    #     inference_model = Inference_SoH_Normal("save_params/trained_model_normal.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
    #     return inference_model
    else:
        raise ValueError("Modo no reconocido. Usa 'onnx' o 'pth'.")

def realizar_inferencia(x_test, y_test, test_loader, modo="onnx", modelo=None):
    """Realiza la inferencia usando ONNX o PyTorch y calcula métricas."""


    if modo == "onnx":
        predicciones = []
        etiquetas_reales = []
        mae_total, mse_sum, mape_total = 0, 0, 0
        modelo, input_name = cargar_modelo(modo, modelo)
        for idx in range(len(x_test)):
            x_sample = x_test[idx].numpy().astype(np.float32)  # Convertir tensor a numpy
            x_sample = np.expand_dims(x_sample, axis=0)  # Añadir batch dimension

            # Inferencia con ONNX
            output = modelo.run(None, {input_name: x_sample})[0]
            # # Inferencia con PyTorch
            # else:
            #     with torch.no_grad():
            #         x_tensor = torch.tensor(x_sample)
            #         output = inference_model.predict(x_tensor)

            # Guardar predicción y etiqueta real
            pred = output[0]  # Asumimos salida en la primera posición
            real = y_test[idx].item()
            predicciones.append(pred)
            etiquetas_reales.append(real)

            # Cálculo de errores
            mae_total += np.abs(pred - real)
            mse_sum += (pred - real) ** 2
            if real != 0:
                mape_total += np.abs((pred - real) / real)

            # Mostrar resultado parcial
            print(f"Ejemplo {idx + 1}/{len(x_test)} -> Predicción: {pred}, Etiqueta Real: {real}")

    if modo == "pth":
        predicciones = []
        etiquetas_reales = []
        mae_total, mse_sum, mape_total, smap_total = 0, 0, 0, 0
        # sand_model = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers)
        # # Cargar los pesos del modelo entrenado
        #checkpoint = torch.load("save_params/trained_model_narx.pth", map_location=device)
        # sand_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # sand_model.to(device)
        # sand_model.eval()
        #
        # with torch.no_grad():
        #     for idx in range(len(x_test)):
        #         x_sample = x_test[idx].clone().detach().to(device)
        #         soh_raw = sand_model(x_sample)  # Obtener SoH
        #         pred = torch.sigmoid(soh_raw).cpu().numpy()
        #         real = y_test[idx].item()
        #         predicciones.append(pred)
        #         etiquetas_reales.append(real)

        #Inference SoH Normal ###############################
        #inference_model = Inference_SoH_Normal("save_params/trained_model_normal.pth", input_features=3, seq_len=400, n_heads=16, factor=1, n_class=1, n_layers=8)
        #inference_model = Inference_SoH_Normal_Improve(modelo, input_features=3, seq_len=400, n_heads=1, factor=1, n_class=1, n_layers=8)
        inference_model = Inference_SoH_NARX(modelo, input_features=3, seq_len=400, n_heads=num_attention, num_cycles = num_cycles, num_preds=num_preds)
        # inference_model = Inference_SoH_Siamese(modelo, input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
        soh_predictions = inference_model.predict(fixed_train_loader)
        # soh_predictions = inference_model.predict(fixed_test_loader)
        #Inference SoH ###############################




        # real = soh_predictions[1]
        real_values = []
        pred_values = []
        x_test = x_train_narx
        # x_test = x_test_narx
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
            smap_total += np.abs(smap_sup / smap_inf)

            # Mostrar resultado parcial
            print(f"Ejemplo {idx + 1}/{len(x_test)} -> Predicción: {pred}, Etiqueta Real: {real}")

        # Graficar los valores reales y predichos
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(real_values[:100])), real_values[:100], label="Real", color="blue", marker="o")
        plt.scatter(range(len(pred_values[:100])), pred_values[:100], label="Predicho", color="red", marker="x")


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
    smape = smap_total / len(x_test)
    # Calcula el MAPE promedio
    mape = mape_total / len(x_test)
    #  Multiplica por 100 para tener el resultado en porcentaje
    # mape_total*= 100
    #mape = (mape_total / len(x_test)) * 100

    print("\nMétricas finales:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"SMAPE: {smape}")


def realizar_inferencia_narx(x_test, y_test, cap_test, modo="onnx", modelo=None):
    """Realiza la inferencia usando ONNX o PyTorch y calcula métricas."""


    if modo == "onnx":
        predicciones = []
        etiquetas_reales = []
        mae_total, mse_sum, mape_total = 0, 0, 0
        modelo, input_name = cargar_modelo(modo, modelo)
        for idx in range(len(x_test_narx)):
            x_sample = x_test_narx[idx].numpy().astype(np.float32)  # Convertir tensor a numpy
            x_sample = np.expand_dims(x_sample, axis=0)  # Añadir batch dimension

            # Inferencia con ONNX
            output = modelo.run(None, {input_name: x_sample})[0]
            # # Inferencia con PyTorch
            # else:
            #     with torch.no_grad():
            #         x_tensor = torch.tensor(x_sample)
            #         output = inference_model.predict(x_tensor)

            # Guardar predicción y etiqueta real
            pred = output[0]  # Asumimos salida en la primera posición
            real = y_test[idx].item()
            predicciones.append(pred)
            etiquetas_reales.append(real)

            # Cálculo de errores
            mae_total += np.abs(pred - real)
            mse_sum += (pred - real) ** 2
            if real != 0:
                mape_total += np.abs((pred - real) / real)

            # Mostrar resultado parcial
            print(f"Ejemplo {idx + 1}/{len(x_test)} -> Predicción: {pred}, Etiqueta Real: {real}")

    if modo == "pth":
        predicciones = []
        etiquetas_reales = []
        mae_total, mse_sum, mape_total, smap_total = 0, 0, 0, 0
        # sand_model = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers)
        # # Cargar los pesos del modelo entrenado
        #checkpoint = torch.load("save_params/trained_model_narx.pth", map_location=device)
        # sand_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # sand_model.to(device)
        # sand_model.eval()
        #
        # with torch.no_grad():
        #     for idx in range(len(x_test)):
        #         x_sample = x_test[idx].clone().detach().to(device)
        #         soh_raw = sand_model(x_sample)  # Obtener SoH
        #         pred = torch.sigmoid(soh_raw).cpu().numpy()
        #         real = y_test[idx].item()
        #         predicciones.append(pred)
        #         etiquetas_reales.append(real)

        #Inference SoH Normal ###############################
        # inference_model = Inference_SoH_Normal("save_params/trained_model_normal.pth", input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
        #inference_model = Inference_SoH_Normal_Improve(modelo, input_features=3, seq_len=400, n_heads=1, factor=1, n_class=1, n_layers=8)
        inference_model = Inference_SoH_NARX(modelo, input_features=feature_dim1, seq_len=feature_dim2, n_heads=num_attention, num_cycles = num_cycles, num_preds=num_preds)
        # inference_model = Inference_SoH_Siamese(modelo, input_features=3, seq_len=400, n_heads=32, factor=32, n_class=1, n_layers=4)
        soh_predictions = inference_model.predict(fixed_test_loader)
        #Inference SoH ###############################




        # real = soh_predictions[1]
        real_values = []
        pred_values = []
        x_test = x_test_narx
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
            smap_total += np.abs(smap_sup / smap_inf)

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
    smape = smap_total / len(x_test)
    # Calcula el MAPE promedio
    mape = mape_total / len(x_test)
    #  Multiplica por 100 para tener el resultado en porcentaje
    # mape_total*= 100
    #mape = (mape_total / len(x_test)) * 100

    print("\nMétricas finales:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"SMAPE: {smape}")


# 🔹 Ejemplo de uso
if inference ==  True:
    modo = "pth"  # Cambia a "pth" para usar el modelo original

    # model = torch.load("save_params/trained_model_narx.pt", weights_only=False)
    # # Extraer los pesos
    # state_dict = model.state_dict()
    # # Guardar solo el state_dict
    # torch.save({"model_state_dict": state_dict}, "save_params/trained_model_narx_new.pth")
    modelo ="save_params/trained_model_narx.pth"
    #realizar_inferencia(x_test, y_test, test_loader, modo, modelo)
    #realizar_inferencia_narx(x_test_narx, y_test_narx, cap_test, modo, modelo)
    realizar_inferencia_narx(x_pairs_fixed_test, cap_inputs_fixed_test, y_targets_fixed_test, modo, modelo)
    #realizar_inferencia("save_params/trained_model_normal.pth")


