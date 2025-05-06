"""
COMENTARIOS GENERALES (una vez leído se puede borrar)

Este script realiza la conversión de las variables de tensión y corriente de pack a celda unitaria, también aplica una interpolación de dichos valores para tener un número de puntos determinado por curva, y los normaliza entre -1 y 1, convirtiéndolos finalmente en formato tensorial.

La función convert_data es quien se encarga de realizar todo el proceso descrito.

También, debajo de dicha función se ha añadido un código para la comprobación del correcto funcionamiento.

"""

"""
Created on Tue Apr 29

@author: acarulla
"""

import numpy as np
import scipy
import torch

def convert_data(N_serial, N_parallel, V, I, SoH):
    """
    Entradas:
        - N_serial: Número de celdas en serie de la batería.
        - N_parallel: Número de celdas en paralelo de la batería.
        - V: Curva de valores de tensión.
        - I: Curva de valores de intensidad.
        - SoH: Etiqueta del estado de salud de la batería.
    Salidas:
        - data_tensor: Tensor con los valores de tensión e intensidad procesados (en columnas).
        - soh_tensor: Tensor del valor del estado de salud de la batería (escalar).
    """
    
    # Se estudia si el número de valores de tensión y corriente coinciden
    if len(V) != len(I):
        raise ValueError(f"El número de valores de tensión ({len(V)}) y corriente ({len(I)}) no coinciden.")
    else: 
        num_samples = len(V)
    
    # Se convierten los valores de pack a celda estándar
    V_cell = np.array(V) / N_serial
    I_cell = np.array(I) / N_parallel
    
    # Se interpola para tener exactamente un determinado número de puntos en todo el rango de carga/descarga
    num_points = 400
    t_original = np.linspace(0, 1, num_samples)
    t_interp = np.linspace(0, 1, num_points)
    V_cell_interp = scipy.interpolate.interp1d(t_original, V_cell, kind='linear')(t_interp)
    I_cell_interp = scipy.interpolate.interp1d(t_original, I_cell, kind='linear')(t_interp)
    
    # Se normalizan los valores en el rango [-1, 1]
    V_cell_interp_norm = 2 * (V_cell_interp - np.min(V_cell_interp)) / (np.max(V_cell_interp) - np.min(V_cell_interp)) - 1
    I_cell_interp_norm = 2 * (I_cell_interp - np.min(I_cell_interp)) / (np.max(I_cell_interp) - np.min(I_cell_interp)) - 1
    
    # Se crea el tensor (número de puntos, 2) y la etiqueta (SoH)
    data_tensor = torch.stack((torch.from_numpy(V_cell_interp_norm).float(), torch.from_numpy(I_cell_interp_norm).float()), dim=1)
    soh_tensor = torch.tensor(SoH)
    
    return data_tensor, soh_tensor


# """ Parte de verificación """
#
# import matplotlib.pyplot as plt
#
# # Generación de valores de tensión y corriente a modo de ejemplo
# V = [x**2 for x in range(1, 1000)]
# I = [-0.8 * x**2 for x in range(1, 1000)]
#
# # Representación de los valores originales
# t_norm = np.linspace(0, 1, len(V))
# plt.figure(figsize=(10, 5))
# plt.plot(t_norm, V, label='Tensión (V)')
# plt.plot(t_norm, I, label='Corriente (A)')
# plt.legend()
# plt.title("Tensión y corriente originales")
# plt.xlabel("Valor temporal (normalizado)")
# plt.ylabel("Valor")
# plt.grid(True)
# plt.show()
#
# # Invocación de la función que realiza el procesamiento de los datos
# data_tensor, soh_tensor = convert_data(4, 4, V, I, 90)
#
# # Representación de los valores procesados
# t_norm = np.linspace(0, 1, data_tensor.shape[0])
# plt.figure(figsize=(10, 5))
# plt.plot(t_norm, data_tensor[:, 0], label='Tensión')
# plt.plot(t_norm, data_tensor[:, 1], label='Corriente')
# plt.legend()
# plt.title("Tensión y corriente normalizadas")
# plt.xlabel("Valor temporal (normalizado)")
# plt.ylabel("Valor normalizado [-1, 1]")
# plt.grid(True)
# plt.show()
#
# print(f"El número de valores de la curva de tensión original es {len(V)}.")
# print(f"El número de valores de la curva de corriente original es {len(I)}.")
# print(f"El número de valores de la curva de tensión procesada es {len(data_tensor[:, 0])}.")
# print(f"El número de valores de la curva de corriente procesada es {len(data_tensor[:, 1])}.")