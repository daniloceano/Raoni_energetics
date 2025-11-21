
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# Pasta de saída
output_dir = "wind_profile_figures"
os.makedirs(output_dir, exist_ok=True)

models = [
    {
        "file": "Raoni_COWAST_merged_processed.nc",
        "name": "COWAST",
        "vars": {"U": "U", "V": "V", "W": "W", "T": "T", "GPH": "GPH"},
        "coords": {"time": "Time", "pressure": "plevels"}
    },
    {
        "file": "Raoni_WRF_merged_processed.nc",
        "name": "WRF",
        "vars": {"U": "U", "V": "V", "W": "W", "T": "T", "GPH": "GPH"},
        "coords": {"time": "Time", "pressure": "plevels"}
    },
    {
        "file": "Raoni_ERA5.nc",
        "name": "ERA5",
        "vars": {"U": "u", "V": "v", "W": "w", "T": "t", "GPH": "z"},
        "coords": {"time": "valid_time", "pressure": "pressure_level"}
    }
]

multiplot_vars = ["U", "V", "W", "T", "GPH"]

def print_model_header(models):
    print("\n" + "="*40)
    print("Modelos:", [m["name"] for m in models])
    print("Arquivos:", [m["file"] for m in models])
    print("Variáveis por modelo:")
    for m in models:
        print(f"  {m['name']}: {list(m['vars'].values())}")
    print("Coordenadas por modelo:")
    for m in models:
        print(f"  {m['name']}: tempo={m['coords']['time']}, pressão={m['coords']['pressure']}")

for var_plot in multiplot_vars:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False, sharey=True)
    if var_plot == multiplot_vars[0]:
        print_model_header(models)
    for col, m in enumerate(models):
        print(f"\nProcessando {m['file']} para {var_plot}")
        ds = xr.open_dataset(m['file'])
        print(f"Variáveis encontradas em {m['name']}: {list(ds.variables)}")
        plevel_name = m['coords']['pressure']
        time_name = m['coords']['time']
        print(f"Eixo vertical usado: {plevel_name}")
        print(f"Eixo temporal usado: {time_name}")
        if plevel_name not in ds.variables or time_name not in ds.variables:
            print(f"Não foi possível encontrar coordenadas de pressão ou tempo em {m['file']}")
            continue
        var_ds = m['vars'][var_plot]
        if var_ds not in ds:
            print(f"Variável {var_plot} não encontrada em {m['file']}")
            continue
        da = ds[var_ds]
        plevels = ds[plevel_name]
        # Seleciona apenas níveis >= 100 hPa
        plevels_mask = plevels >= 100
        da = da.sel({plevel_name: plevels[plevels_mask]})
        plevels = plevels[plevels_mask]
        spatial_dims = [d for d in da.dims if d not in [time_name, plevel_name]]
        # Média
        zonal_mean = da.mean(dim=spatial_dims)
        ax = axes[0, col]
        if plevels.size > 0 and zonal_mean.size > 0:
            mesh = ax.pcolormesh(ds[time_name], plevels, zonal_mean.T, shading="auto")
            ax.set_title(f"{m['name']} - Média")
            ax.set_ylabel("Nível de pressão (hPa)")
            ax.invert_yaxis()
            plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.01)
        else:
            ax.set_title(f"{m['name']} - Média\nSem dados válidos")
        # Máximo
        max_profile = da.max(dim=spatial_dims)
        ax = axes[1, col]
        if plevels.size > 0 and max_profile.size > 0:
            mesh = ax.pcolormesh(ds[time_name], plevels, max_profile.T, shading="auto")
            ax.set_title(f"{m['name']} - Máximo")
            ax.set_ylabel("Nível de pressão (hPa)")
            ax.invert_yaxis()
            plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.01)
        else:
            ax.set_title(f"{m['name']} - Máximo\nSem dados válidos")
        # Mínimo
        min_profile = da.min(dim=spatial_dims)
        ax = axes[2, col]
        if plevels.size > 0 and min_profile.size > 0:
            mesh = ax.pcolormesh(ds[time_name], plevels, min_profile.T, shading="auto")
            ax.set_title(f"{m['name']} - Mínimo")
            ax.set_xlabel("Tempo")
            ax.set_ylabel("Nível de pressão (hPa)")
            ax.invert_yaxis()
            plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.01)
        else:
            ax.set_title(f"{m['name']} - Mínimo\nSem dados válidos")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hovmoller_multiplot_{var_plot}.png"))
    plt.close()
    print(f"Figura multiplot salva: hovmoller_multiplot_{var_plot}.png")
