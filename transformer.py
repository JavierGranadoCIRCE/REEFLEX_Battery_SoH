from comet_ml import Experiment
import torch.nn as nn
import torch.optim as optim
import random
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

# Normaliza canal a canal
means = data.mean(dim=(0, 1), keepdim=True)  # media por canal
stds = data.std(dim=(0, 1), keepdim=True)    # std por canal

data = (data - means) / stds  # normalización global por canal

# Guardar para uso posterior
torch.save({'mean': means, 'std': stds}, 'save_params/normalization_stats.pt')


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

##########################################################################
# PLoteo de los ciclos
##########################################################################
#ploteo_NARX(val_ds_narx)
#ploteo_NN4SOH_aaptado_a_NARX(train_dataset)
#ploteo_NN4SOH(x_train, y_train)
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
    train = True
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
    save_example_to_csv_narx(x_test_narx, cap_test, y_test_narx, 249, filename="save_params/ciclo_de_carga_narx_250.csv")
################################################################################



#################################################################################################################3
# Training process
###############################################################################################################

if train == True:

    if finetuning == True:
        modelo, checkpoint = cargar_modelo_pth_finetuning(NARX_Transformer,"save_params/trained_model_narx_2var_tripletes_random.pth")
        modelo = modelo.to(device)
        epoch = checkpoint['epoch']
        optimizer = torch.optim.Adam(modelo.parameters())  # Usando el lr guardado
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        clf.fit_NARX_Transformer_finetuning(epoch, optimizer, modelo, x_pairs_fixed_train, y_targets_fixed_train, x_pairs_fixed_val, y_targets_fixed_val, x_pairs_fixed_test, y_targets_fixed_test,
                                 {"train_narx": fixed_train_loader,
                                  "val_narx": fixed_val_loader,
                                  "test_narx": fixed_test_loader},
                                 epochs=1000
                                 )
        # ##################################################################################################
        # # # save pth model when finetuning process is completed
        # #############################################################################################
        clf.save_to_file_Narx_finetuning("trained_model_narx_2var_1ciclo.pth")
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
                epochs=300
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
    modelo ="save_params/trained_model_narx_2var_1ciclo.pth"
    realizar_inferencia_narx(test_loader, x_test, y_test, modo, modelo)
    #realizar_inferencia_narx(test_loader_narx, x_test_narx, cap_test, y_test_narx, modo, modelo)
    #realizar_inferencia_narx(fixed_test_loader,x_pairs_fixed_test, cap_inputs_fixed_test, y_targets_fixed_test, modo, modelo)



