
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modelos e pastas
modelos = [
    ("COWAST_merged_processed_fixed", "COWAST"),
    ("WRF_merged_processed_fixed", "WRF"),
    ("ERA5_fixed", "ERA5")
]


# Arquivos de cada linha para cada modelo
ck_files_by_model = [
    [f"Ck_{i}_plevels.csv" for i in range(1, 6)],  # COWAST
    [f"Ck_{i}_plevels.csv" for i in range(1, 6)],  # WRF
    [f"Ck_{i}_pressure_level.csv" for i in range(1, 6)]  # ERA5
]

# Pasta base
base_dir = os.path.dirname(__file__)

fig, axes = plt.subplots(5, 3, figsize=(15, 12), sharex=True, sharey=True)

for col, (folder, model_name) in enumerate(modelos):
    ck_files = ck_files_by_model[col]
    for row, ck_file in enumerate(ck_files):
        file_path = os.path.join(base_dir, "LEC_Results", f"Raoni_{folder}", "results_vertical_levels", ck_file)
        ax = axes[row, col]
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=0)
            # Sempre: primeira coluna é tempo, primeira linha (exceto o primeiro valor) são níveis de pressão
            plevels = np.array([float(p) for p in df.columns[1:]])
            times = df.iloc[:, 0].values
            data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values  # shape: (tempo, pressão)
            # Converter Pa para hPa se necessário
            if plevels.max() > 2000:
                plevels = plevels / 100.0
            # Se todos os dados forem NaN, não plota
            if np.isnan(data).all():
                ax.set_title(f"{model_name} - {ck_file}\nDados não encontrados ou inválidos", fontsize=8)
            else:
                mesh = ax.pcolormesh(times, plevels, data.T, shading="auto")
                ax.set_title(f"{model_name} - {ck_file}", fontsize=10)
                ax.invert_yaxis()
                plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.01)
                # Ajusta os ticks do eixo y para não sobrepor
                n_ticks = 6
                plevel_ticks = np.linspace(plevels.min(), plevels.max(), n_ticks)
                ax.set_yticks(plevel_ticks)
                ax.set_yticklabels([f"{int(t)}" for t in plevel_ticks])
                # Rotaciona os ticks do eixo x
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.set_title(f"{model_name} - {ck_file}\nArquivo não encontrado", fontsize=8)
        if col == 0:
            ax.set_ylabel("Pressão (hPa)")
        if row == 4:
            ax.set_xlabel("Tempo")

plt.tight_layout()
plt.savefig("Ck_plevels_comparacao.png")
plt.close()
print("Figura salva: Ck_plevels_comparacao.png")
