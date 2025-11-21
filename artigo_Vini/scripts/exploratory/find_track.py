import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import timedelta

# Caminho do arquivo
tracks_path = '../../energetic_patterns_cyclones_south_atlantic/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'

# Carregar os dados
tracks = pd.read_csv(tracks_path)

# Converter a coluna 'date' para datetime
tracks['date'] = pd.to_datetime(tracks['date'])

# Definir os critérios do ciclone Raoni
start_date = pd.to_datetime("2021-06-26 18:00:00") - timedelta(days=2)
end_date = pd.to_datetime("2021-07-01 18:00:00") + timedelta(days=2)
region_lon_min, region_lon_max = -65, -30
region_lat_min, region_lat_max = -50, -30

# Filtrar os dados
raoni_tracks = tracks[
    (tracks['date'] >= start_date) &
    (tracks['date'] <= end_date) &
    (tracks['lon vor'] >= region_lon_min) &
    (tracks['lon vor'] <= region_lon_max) &
    (tracks['lat vor'] >= region_lat_min) &
    (tracks['lat vor'] <= region_lat_max)
]

# Identificar o track_id com mais pontos
if not raoni_tracks.empty:
    track_id = raoni_tracks['track_id'].value_counts().idxmax()
    raoni_track = raoni_tracks[raoni_tracks['track_id'] == track_id]

    # Plotar o mapa
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 8))
    ax.set_extent([region_lon_min, region_lon_max, region_lat_min, region_lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plotar a trajetória
    ax.plot(
        raoni_track['lon vor'], 
        raoni_track['lat vor'], 
        marker='o', color='black', label=f'Track ID {track_id}'
    )

    # Adicionar anotações para início e fim
    ax.text(
        raoni_track.iloc[0]['lon vor'], raoni_track.iloc[0]['lat vor'],
        'A', fontsize=12, fontweight='bold', color='black', transform=ccrs.PlateCarree()
    )
    ax.text(
        raoni_track.iloc[-1]['lon vor'], raoni_track.iloc[-1]['lat vor'],
        'Z', fontsize=12, fontweight='bold', color='black', transform=ccrs.PlateCarree()
    )

    # Título e legenda
    ax.set_title('Cyclone Raoni Track', fontsize=14)
    ax.legend()

    plt.show()
else:
    print("Nenhum ciclone encontrado que atenda aos critérios especificados.")
