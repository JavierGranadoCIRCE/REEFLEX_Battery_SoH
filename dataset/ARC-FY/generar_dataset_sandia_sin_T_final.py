import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import mat_struct
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.ndimage import binary_dilation
from scipy.signal import medfilt
import os

# Ruta y archivos
ruta = "C:/Users/reeflex/olimpIAdas_VoltIA/dataset/ARC-FY"
archivos = ["NMC_CELL_1.mat", "NMC_CELL_2.mat"]

def suavizar(x, kernel_size=5):
    return medfilt(x, kernel_size=kernel_size)

def normalizar(x):
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1 if np.max(x) != np.min(x) else x * 0

def extraer_segmento(t, I, V):
    I_norm = normalizar(I)
    V_norm = normalizar(V)
    try:
        inicio = np.argmax(I_norm > 0.8)
        fin_candidates = np.where((I_norm < 0.68) & (V_norm > 0.95) & (np.arange(len(I)) > inicio))[0]
        if len(fin_candidates) == 0:
            return None
        fin = fin_candidates[0]
        if fin - inicio < 10:
            return None
        return t[inicio:fin], I[inicio:fin], V[inicio:fin]
    except:
        return None

ciclos_procesados = []

for archivo in archivos:
    print(f"🔍 Explorando archivo: {archivo}")
    data = scipy.io.loadmat(os.path.join(ruta, archivo), struct_as_record=False, squeeze_me=True)
    tabla = data["table"]

    for i, ciclo in enumerate(tabla):
        try:
            if archivo == "NMC_CELL_2.mat" and i == 2:
                print(f"⚠️ Ciclo {i} saltado manualmente por ser erróneo.")
                continue

            I = np.array(ciclo.Current).flatten()
            V = np.array(ciclo.Voltage).flatten()
            if len(I) != len(V):
                print(f"⚠️ Ciclo {i} descartado: dimensiones no coinciden I:{len(I)} V:{len(V)}")
                continue

            t = np.arange(len(V))
            print(f"🔍 Plot ciclo {i} — t:{t.shape}, I:{I.shape}, V:{V.shape}")
            segmento = extraer_segmento(t, I, V)

            if segmento is None:
                print(f"⚠️ Ciclo {i} descartado: segmento no válido")
                continue

            t_rec, I_rec, V_rec = segmento
            t_rec = t_rec - t_rec[0]

            # Interpolación y normalización
            f_I = interp1d(t_rec, I_rec, kind="linear")
            f_V = interp1d(t_rec, V_rec, kind="linear")
            t_uniforme = np.linspace(0, t_rec[-1], 400)
            I_interp = normalizar(f_I(t_uniforme))
            V_interp = normalizar(f_V(t_uniforme))

            # Eliminar escalones en la corriente y corrección en la tensión
            mask_corriente_salto = I_interp == -1
            if np.any(mask_corriente_salto):
                I_interp[mask_corriente_salto] = 1
                mascara_expandida = binary_dilation(mask_corriente_salto, iterations=2)
                V_interp[mascara_expandida] = 1

            # Suavizado de picos residuales
            I_interp = suavizar(I_interp, kernel_size=5)
            V_interp = suavizar(V_interp, kernel_size=5)

            # Cálculo de capacidad y SoH
            I_rec_amperios = I_rec / 1000  # convertir de mA a A
            Q = trapezoid(I_rec_amperios, t_rec)  # en A·s
            Q_ah = Q / 3600  # pasar a Ah
            Q_nominal = 3.0
            SoH = Q_ah / Q_nominal

            # 🔍 Plot después del recorte y limpieza
            plt.figure(figsize=(10, 3))
            plt.plot(np.arange(400), I_interp, label='Corriente')
            plt.plot(np.arange(400), V_interp, label='Tensión')
            plt.title(f'Ciclo {i} — SoH: {SoH:.3f}')
            plt.xlabel('Tiempo')
            plt.ylabel('Magnitud')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            entrada = np.stack([V_interp, I_interp], axis=-1)
            ciclos_procesados.append((entrada, SoH))

        except Exception as e:
            print(f"⚠️ Ciclo {i} descartado: {e}")

# Guardado final
if len(ciclos_procesados) == 0:
    print("❌ No se han encontrado ciclos válidos.")
else:
    X = np.array([x[0] for x in ciclos_procesados])
    y = np.array([x[1] for x in ciclos_procesados])
    dataset = np.concatenate([X.reshape(X.shape[0], -1), y[:, None]], axis=1)
    output_path = os.path.join(ruta, "dataset_final.mat")
    scipy.io.savemat(output_path, {"dataset": dataset})
    print(f"💾 Dataset guardado como dataset_final.mat con forma {dataset.shape}")


