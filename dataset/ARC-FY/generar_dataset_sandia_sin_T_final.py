import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import os
from scipy.ndimage import binary_dilation
from scipy.signal import medfilt

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
        return t[inicio:fin], I[inicio:fin], V[inicio:fin], inicio, fin
    except:
        return None

ciclos_procesados = []

for archivo in archivos:
    print(f"🔍 Explorando archivo: {archivo}")
    data = scipy.io.loadmat(os.path.join(ruta, archivo), struct_as_record=False, squeeze_me=True)
    tabla = data["table"]

    for i, ciclo in enumerate(tabla):
        try:
            I = np.array(ciclo.Current).flatten()
            V = np.array(ciclo.Voltage).flatten()

            if len(I) != len(V):
                print(f"⚠️ Ciclo {i} descartado: dimensiones no coinciden I:{len(I)} V:{len(V)}")
                continue

            if archivo == "NMC_CELL_2.mat" and i == 2:
                print(f"⚠️ Ciclo {i} descartado manualmente por ser anómalo")
                continue

            t = np.arange(len(V))
            print(f"🔍 Plot ciclo {i} — t:{t.shape}, I:{I.shape}, V:{V.shape}")

            I_original = I.copy()  # Guardamos corriente original antes de modificar

            resultado = extraer_segmento(t, I, V)
            if resultado is None:
                print(f"⚠️ Ciclo {i} descartado: segmento no válido")
                continue

            t_rec, I_rec, V_rec, inicio, fin = resultado
            t_rec = t_rec - t_rec[0]

            # Cálculo de capacidad con corriente original (sin normalizar)
            I_rec_amperios = I_original[inicio:fin]
            Q = trapezoid(I_rec_amperios, t_rec) / 3600  # Capacidad en Ah
            corriente_media = np.mean(I_rec_amperios)
            print(f"Ciclo {i} — duración del segmento: {t_rec[-1]:.2f}s, corriente media: {corriente_media:.4f} A")

            f_I = interp1d(t_rec, I_rec, kind="linear")
            f_V = interp1d(t_rec, V_rec, kind="linear")
            t_uniforme = np.linspace(0, t_rec[-1], 400)
            I_interp = normalizar(f_I(t_uniforme))
            V_interp = normalizar(f_V(t_uniforme))

            # Eliminar escalones
            mask_corriente_salto = I_interp == -1
            if np.any(mask_corriente_salto):
                I_interp[mask_corriente_salto] = 1
                mascara_expandida = binary_dilation(mask_corriente_salto, iterations=2)
                V_interp[mascara_expandida] = 1

            # Suavizar picos
            I_interp = suavizar(I_interp, kernel_size=5)
            V_interp = suavizar(V_interp, kernel_size=5)

            # Cálculo del SoH
            Q_nominal = 4
            SoH = Q / Q_nominal

            # Plot final con SoH
            # plt.figure(figsize=(10, 3))
            # plt.plot(np.arange(400), I_interp, label='Corriente')
            # plt.plot(np.arange(400), V_interp, label='Tensión')
            # plt.title(f'SoH = {SoH:.3f}')
            # plt.xlabel('Tiempo')
            # plt.ylabel('Magnitud')
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()

            entrada = np.stack([I_interp, V_interp], axis=-1)
            ciclos_procesados.append((entrada, SoH))

        except Exception as e:
            print(f"⚠️ Ciclo {i} descartado: {e}")

if len(ciclos_procesados) == 0:
    print("❌ No se han encontrado ciclos válidos.")
else:
    X = np.array([x[0] for x in ciclos_procesados])  # (num_samples, 400, 2)
    y = np.array([x[1] for x in ciclos_procesados])  # (num_samples,)
    dataset = np.concatenate([X.reshape(X.shape[0], -1), y[:, None]], axis=1)
    output_path = os.path.join(ruta, "dataset_final.mat")
    scipy.io.savemat(
        output_path,
        {"dataset": np.array(dataset, dtype=np.float32)},
        do_compression=True,
        long_field_names=True
    )
    print(f"📂 Dataset guardado como dataset_final.mat con forma {dataset.shape}")

