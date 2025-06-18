
import scipy.io
import numpy as np
from scipy.interpolate import interp1d
import os

# Cargar el archivo original
mat = scipy.io.loadmat("dataset/ARC-FY/NMC_CELL_1.mat", struct_as_record=False, squeeze_me=True)
tabla = mat["table"]

resultados = []
ciclos_procesados = 0
ciclos_descartados = 0

for fila in tabla:
    try:
        V = fila.Voltage.astype(np.float32)
        I = fila.Current.astype(np.float32)
        T = fila.Temperature.astype(np.float32)

        # Detectar tramo de carga: corriente positiva
        umbral_carga = 1.0
        idx_carga = np.where(I > umbral_carga)[0]

        if len(idx_carga) < 50:
            ciclos_descartados += 1
            continue

        # Buscar tramo continuo
        difs = np.diff(idx_carga)
        cortes = np.where(difs > 1)[0]
        if len(cortes) > 0:
            idx_carga = idx_carga[:cortes[0]]

        # Recortar señales
        V_c = V[idx_carga]
        I_c = I[idx_carga]
        T_c = T[idx_carga]

        if min(len(V_c), len(I_c), len(T_c)) < 50:
            ciclos_descartados += 1
            continue

        # Interpolación a 400 puntos
        x_old = np.linspace(0, 1, num=len(V_c))
        x_new = np.linspace(0, 1, num=400)

        V_interp = interp1d(x_old, V_c)(x_new)
        I_interp = interp1d(x_old, I_c)(x_new)
        T_interp = interp1d(x_old, T_c)(x_new)

        # Calcular SoH como capacidad relativa al primer ciclo
        Q = np.abs(np.trapz(I_c, dx=1))
        SoH = Q

        if SoH <= 0:
            ciclos_descartados += 1
            continue

        if ciclos_procesados == 0:
            Q_ref = SoH
        SoH_rel = SoH / Q_ref

        resultado = np.concatenate([V_interp, I_interp, T_interp, [SoH_rel]])
        resultados.append(resultado)
        ciclos_procesados += 1
    except Exception:
        ciclos_descartados += 1
        continue

print(f"✅ Ciclos procesados: {ciclos_procesados}")
print(f"❌ Ciclos descartados: {ciclos_descartados}")

if ciclos_procesados > 0:
    data = np.stack(resultados)
    scipy.io.savemat("dataset/ARC-FY/NMC_CELL_1_charge_only.mat", {"data": data})
    print("💾 Guardado como NMC_CELL_1_charge_only.mat")
else:
    print("⚠️ No se han podido procesar ciclos válidos.")
