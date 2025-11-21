import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Abrir o dataset filtrado
ds = xr.open_dataset("Roani_WRF_filtered.nc")

# Selecionar o primeiro instante de tempo e nível de 1000 hPa
ds_1000hPa = ds.sel(plevels=50, method="nearest").isel(Time=0)

# Definir os limites do gráfico com base nos dados (converter para float)
lat_min, lat_max = float(ds_1000hPa.lat.min()), float(ds_1000hPa.lat.max())
lon_min, lon_max = float(ds_1000hPa.lon.min()), float(ds_1000hPa.lon.max())

# Variáveis a serem plotadas
variables = ["T", "QVAPOR", "U", "V", "W", "GPH"]

# Criar mapas para cada variável
for var in variables:
    plt.figure(figsize=(10, 6))
    
    # Criar o mapa com projeção geográfica
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Adicionar features como linhas de costa e fronteiras
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")

    # Plotar os dados como um mapa de cores
    mesh = ax.pcolormesh(ds_1000hPa.lon, ds_1000hPa.lat, ds_1000hPa[var], 
                         transform=ccrs.PlateCarree(), cmap="viridis")

    # Adicionar barra de cores
    cbar = plt.colorbar(mesh, orientation="vertical", pad=0.02)
    cbar.set_label(f"{var} ({ds_1000hPa[var].attrs.get('units', '')})")

    # Título
    plt.title(f"{var} em 50 hPa - Primeiro Instante de Tempo")

    # Mostrar o gráfico
    plt.show()
