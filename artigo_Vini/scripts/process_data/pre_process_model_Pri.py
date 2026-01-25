"""
Script para pré-processar dados de modelos atmosféricos para cálculos de energética.

Este script lê dados de modelos com algum nível de processamento e retorna um arquivo
processado pronto para utilizar nos cálculos de energética. O processamento inclui:
- Seleção de variáveis necessárias
- Conversão de unidades (quando necessário)
- Adição de metadados (unidades)
- Filtragem de níveis verticais
- Conversão de geopotencial para altura geopotencial (quando aplicável)

Autor: Danilo
Data: 2025
"""

from token import NAME
import xarray as xr
import numpy as np
import pandas as pd
import sys

# ============================================================================
# CONFIGURAÇÕES - MODIFIQUE AQUI
# ============================================================================

# Arquivos de entrada e saída
INFILE = 'data/WRFsa-ERA5-RAONI-6h_INTRP-Regular.nc'
OUTFILE = f"{INFILE.replace('.nc', '_processed.nc')}"

# Variáveis a manter no arquivo (lista)
# Use os nomes exatos como aparecem no arquivo WRF
VARS_TO_KEEP = ['TT', 'UU', 'VV', 'W', 'GHT']

# Unidades das variáveis (dicionário: variável -> unidade)
UNITS_DICT = {
    'TT': 'K',           # Temperatura
    'UU': 'm/s',         # Componente zonal do vento
    'VV': 'm/s',         # Componente meridional do vento
    'W': 'Pa/s',         # Velocidade vertical (omega)
    'GHT': 'm',          # Altura geopotencial
}

# Configuração de geopotencial
# Opções: 'geopotencial' (m²/s², será convertido), 'altura' (m, não converte), 'nenhum'
GPH_FLAG = 'altura'
GPH_VAR = 'GHT'  # Nome da variável de geopotencial no arquivo

# Variável omega (velocidade vertical)
# Se quiser multiplicar por -1, deixe como True; se não, False
OMEGA_VAR = 'W'  # Nome da variável omega
MULTIPLY_OMEGA_BY_MINUS_ONE = True  # True para multiplicar por -1, False para não multiplicar

# Níveis de pressão (em hPa)
PLEVEL_MIN = 100  # Nível mínimo (hPa)
PLEVEL_MAX = 1000  # Nível máximo (hPa)
PLEVEL_VAR = 'LEV'  # Nome da variável de níveis de pressão no arquivo WRF
PLEVEL_DIM = 'vlevs'  # Nome da dimensão de níveis verticais

# Dimensões espaciais (serão renomeadas)
LAT_DIM = 'south_north'  # Nome da dimensão de latitude
LON_DIM = 'west_east'    # Nome da dimensão de longitude
LAT_VAR = 'lat'          # Nome da variável de latitude
LON_VAR = 'lon'          # Nome da variável de longitude

# Variáveis de tempo
TIME_VAR = 'time'  # Nome da dimensão/coordenada de tempo principal
TIMES_VAR = None   # Não há variável auxiliar Times neste arquivo WRF
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'  # Formato do tempo

# Coordenadas a remover (lista)
# Deixe como None ou [] para não remover nenhuma coordenada extra
COORDS_TO_REMOVE = []

# Compressão do arquivo de saída
USE_COMPRESSION = True

# Listar variáveis disponíveis e sair (para debug)
LIST_VARS_ONLY = False

# Gerar plots das variáveis (snapshot do meio do período)
GENERATE_PLOTS = True  # True para gerar plots, False para não gerar
PLOT_PLEVEL_INDEX = 3  # Índice do nível de pressão para plotar (0-18, onde 9 ~ 500 hPa)

# ============================================================================
# FIM DAS CONFIGURAÇÕES
# ============================================================================

# Emojis imersivos para feedback
EMOJI_START = "🚀"
EMOJI_OK = "✅"
EMOJI_FAIL = "❌"
EMOJI_INFO = "🔎"
EMOJI_PAPER = "📄"
EMOJI_CONFIG = "⚙️"


def print_available_variables(ds):
    """Imprime as variáveis disponíveis no dataset com sigla -> long_name (quando disponível)."""
    available_vars = list(ds.variables.keys())
    if not available_vars:
        print(f"{EMOJI_INFO} Nenhuma variável encontrada no dataset.")
        return

    print(f"\n{EMOJI_INFO} Variáveis disponíveis no arquivo (sigla -> long_name):")
    for v in available_vars:
        # Tenta obter o atributo long_name; se não existir, tenta standard_name; senão vazio
        attrs = ds[v].attrs if hasattr(ds[v], 'attrs') else {}
        long_name = attrs.get('long_name') or attrs.get('standard_name') or ''
        descriptions = attrs.get('description') or ''
        if long_name:
            print(f"{EMOJI_PAPER} {v} -> {long_name}")
        elif descriptions:
            print(f"{EMOJI_PAPER} {v} -> {descriptions}")
        else:
            print(f"{EMOJI_PAPER} {v}")


# ============================================================================
# INÍCIO DO PROCESSAMENTO
# ============================================================================

print(f"{EMOJI_START} Iniciando pré-processamento: abrindo '{INFILE}'")

# Abrir o dataset
try:
    ds = xr.open_dataset(INFILE)
except Exception as e:
    print(f"{EMOJI_FAIL} Erro ao abrir o arquivo: {e}")
    sys.exit(1)

# Se flag LIST_VARS_ONLY está ativa, apenas listar variáveis e sair
if LIST_VARS_ONLY:
    print_available_variables(ds)
    sys.exit(0)

# Adicionar variáveis auxiliares necessárias
vars_needed = VARS_TO_KEEP.copy()
if LAT_VAR not in vars_needed and LAT_VAR in ds:
    vars_needed.append(LAT_VAR)
if LON_VAR not in vars_needed and LON_VAR in ds:
    vars_needed.append(LON_VAR)
if PLEVEL_VAR not in vars_needed and PLEVEL_VAR in ds:
    vars_needed.append(PLEVEL_VAR)

print(f"\n{EMOJI_CONFIG} Configuração:")
print(f"  Variáveis principais: {', '.join(VARS_TO_KEEP)}")
print(f"  Dimensão vertical: {PLEVEL_DIM}")
print(f"  Variável de pressão: {PLEVEL_VAR}")
print(f"  Unidades: {UNITS_DICT}")
print(f"  Flag geopotencial: {GPH_FLAG}")
if GPH_FLAG != 'nenhum':
    print(f"  Variável geopotencial: {GPH_VAR}")
if OMEGA_VAR:
    print(f"  Variável omega: {OMEGA_VAR} (multiplicar por -1: {MULTIPLY_OMEGA_BY_MINUS_ONE})")
print(f"  Níveis de pressão: {PLEVEL_MIN} - {PLEVEL_MAX} hPa")
print(f"  Gerar plots: {GENERATE_PLOTS}")

# Verificar se todas as variáveis existem no dataset
available = set(ds.variables.keys())
missing = [v for v in vars_needed if v not in available]
if missing:
    print(f"\n{EMOJI_FAIL} Variável(s) não encontrada(s): {', '.join(missing)}")
    print_available_variables(ds)
    print(f"\n{EMOJI_FAIL} Abortando execução. Por favor, ajuste VARS_TO_KEEP com base nas variáveis disponíveis.")
    sys.exit(1)

# Selecionar variáveis
print(f"\n{EMOJI_INFO} Selecionando variáveis...")
ds_filtered = ds[vars_needed]

# Processar níveis de pressão
if PLEVEL_VAR in ds_filtered:
    print(f"{EMOJI_INFO} Processando níveis de pressão...")
    
    # LEV está em Pa, converter para hPa e criar coordenada
    plevels_values = ds_filtered[PLEVEL_VAR].values / 100.0  # Pa para hPa
    
    # Criar coordenada de pressão
    ds_filtered = ds_filtered.assign_coords({PLEVEL_DIM: plevels_values})
    
    # Renomear a dimensão para 'plevels'
    ds_filtered = ds_filtered.rename({PLEVEL_DIM: 'plevels'})
    
    # Adicionar atributos à nova coordenada
    ds_filtered['plevels'].attrs['long_name'] = 'Pressure levels'
    ds_filtered['plevels'].attrs['units'] = 'hPa'
    ds_filtered['plevels'].attrs['positive'] = 'down'
    
    # Filtrar níveis de pressão
    print(f"{EMOJI_INFO} Filtrando níveis de pressão: {PLEVEL_MIN} - {PLEVEL_MAX} hPa")
    try:
        ds_filtered = ds_filtered.sel(plevels=slice(PLEVEL_MAX, PLEVEL_MIN))
    except Exception as e:
        print(f"{EMOJI_INFO} Aviso: não foi possível filtrar níveis de pressão: {e}")
    
    # Remover a variável LEV original, mantendo apenas a coordenada
    if PLEVEL_VAR in ds_filtered.data_vars:
        ds_filtered = ds_filtered.drop_vars(PLEVEL_VAR)

# Processar coordenadas espaciais
print(f"{EMOJI_INFO} Processando coordenadas espaciais...")

# Criar índices numéricos para as dimensões espaciais primeiro
if LAT_DIM in ds_filtered.dims and LON_DIM in ds_filtered.dims:
    # Obter valores de lat e lon antes de renomear
    if LAT_VAR in ds_filtered:
        lat_values = ds_filtered[LAT_VAR].values
    if LON_VAR in ds_filtered:
        lon_values = ds_filtered[LON_VAR].values
    
    # Renomear dimensões para nomes padrão
    ds_filtered = ds_filtered.rename({
        LAT_DIM: 'latitude',
        LON_DIM: 'longitude'
    })
    
    # Atribuir valores de coordenadas às dimensões
    if LAT_VAR in ds_filtered:
        ds_filtered = ds_filtered.assign_coords(latitude=lat_values)
        ds_filtered['latitude'].attrs['long_name'] = 'Latitude'
        ds_filtered['latitude'].attrs['units'] = 'degrees_north'
        ds_filtered['latitude'].attrs['axis'] = 'Y'
        # Remover variável lat duplicada
        if LAT_VAR in ds_filtered.data_vars:
            ds_filtered = ds_filtered.drop_vars(LAT_VAR)
        print(f"{EMOJI_INFO} Coordenada de latitude configurada")
    
    if LON_VAR in ds_filtered:
        ds_filtered = ds_filtered.assign_coords(longitude=lon_values)
        ds_filtered['longitude'].attrs['long_name'] = 'Longitude'
        ds_filtered['longitude'].attrs['units'] = 'degrees_east'
        ds_filtered['longitude'].attrs['axis'] = 'X'
        # Remover variável lon duplicada
        if LON_VAR in ds_filtered.data_vars:
            ds_filtered = ds_filtered.drop_vars(LON_VAR)
        print(f"{EMOJI_INFO} Coordenada de longitude configurada")
    
    print(f"{EMOJI_INFO} Dimensões renomeadas: {LAT_DIM} -> latitude, {LON_DIM} -> longitude")

# Garantir que 'time' seja uma coordenada indexada
if TIME_VAR not in ds_filtered.coords and TIME_VAR in ds:
    print(f"{EMOJI_INFO} Adicionando {TIME_VAR} como coordenada...")
    ds_filtered = ds_filtered.assign_coords({TIME_VAR: ds[TIME_VAR]})

# Converter 'Times' para datetime, se existir
if TIMES_VAR and TIMES_VAR in ds:
    print(f"{EMOJI_INFO} Convertendo {TIMES_VAR} para datetime...")
    try:
        times_str = ds[TIMES_VAR].astype(str)
        times_dt = pd.to_datetime(
            [t.decode('utf-8') if isinstance(t, bytes) else t for t in ds[TIMES_VAR].values],
            format=TIME_FORMAT
        )
        ds_filtered = ds_filtered.assign_coords({TIME_VAR: (TIME_VAR, times_dt)})
        
        # Manter Times como coordenada auxiliar
        if TIMES_VAR not in ds_filtered.coords:
            ds_filtered = ds_filtered.assign_coords({TIMES_VAR: ds[TIMES_VAR]})
    except Exception as e:
        print(f"{EMOJI_INFO} Aviso: não foi possível converter {TIMES_VAR}: {e}")

# Manter apenas as variáveis principais (remover variáveis auxiliares)
print(f"{EMOJI_INFO} Mantendo apenas variáveis principais...")
vars_to_keep_final = [v for v in VARS_TO_KEEP if v in ds_filtered.data_vars]
coords_to_keep = ['time', 'plevels', 'latitude', 'longitude']
coords_to_keep = [c for c in coords_to_keep if c in ds_filtered.coords]

# Criar dataset final apenas com variáveis e coordenadas necessárias
ds_final = ds_filtered[vars_to_keep_final]
for coord in coords_to_keep:
    if coord in ds_filtered.coords and coord not in ds_final.coords:
        ds_final = ds_final.assign_coords({coord: ds_filtered[coord]})

ds_filtered = ds_final

# Remover coordenadas especificadas pelo usuário
if COORDS_TO_REMOVE:
    for coord in COORDS_TO_REMOVE:
        if coord in ds_filtered.coords:
            print(f"{EMOJI_INFO} Removendo coordenada: {coord}")
            try:
                del ds_filtered.coords[coord]
            except Exception as e:
                print(f"{EMOJI_INFO} Aviso: não foi possível remover {coord}: {e}")

# Renomear variáveis inválidas (numéricas)
rename_dict = {var: f"var_{var}" for var in ds_filtered.data_vars if isinstance(var, (int, float))}
if rename_dict:
    print(f"{EMOJI_INFO} Renomeando variáveis inválidas: {rename_dict}")
    ds_filtered = ds_filtered.rename(rename_dict)

# Converter todas as variáveis para float32
print(f"{EMOJI_INFO} Convertendo variáveis para float32...")
for var in ds_filtered.data_vars:
    ds_filtered[var] = ds_filtered[var].astype(np.float32)

# Processar geopotencial, se aplicável
if GPH_FLAG == 'geopotencial' and GPH_VAR in ds_filtered:
    print(f"{EMOJI_INFO} Convertendo geopotencial para altura geopotencial (dividindo por 9.81)...")
    ds_filtered[GPH_VAR] = ds_filtered[GPH_VAR] / 9.81
    # Atualizar unidade
    if GPH_VAR in UNITS_DICT and UNITS_DICT[GPH_VAR] in ['m²/s²', 'm2/s2', 'm^2/s^2']:
        UNITS_DICT[GPH_VAR] = 'm'
elif GPH_FLAG == 'altura':
    print(f"{EMOJI_INFO} Variável {GPH_VAR} já está em altura geopotencial (m)")

# Multiplicar omega por -1, se aplicável
if OMEGA_VAR and OMEGA_VAR in ds_filtered and MULTIPLY_OMEGA_BY_MINUS_ONE:
    print(f"{EMOJI_INFO} Multiplicando {OMEGA_VAR} por -1...")
    ds_filtered[OMEGA_VAR] = ds_filtered[OMEGA_VAR] * -1
elif OMEGA_VAR and OMEGA_VAR in ds_filtered:
    print(f"{EMOJI_INFO} Mantendo {OMEGA_VAR} com sinal original (não multiplicado por -1)")

# Adicionar unidades às variáveis
if UNITS_DICT:
    print(f"{EMOJI_INFO} Adicionando unidades às variáveis...")
    for var, unit in UNITS_DICT.items():
        if var in ds_filtered:
            ds_filtered[var].attrs["units"] = unit

# Gerar plots das variáveis (snapshot)
if GENERATE_PLOTS:
    print(f"\n{EMOJI_INFO} Gerando plots das variáveis (snapshot do meio do período)...")
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.gridspec import GridSpec
        
        # Selecionar tempo do meio
        n_times = len(ds_filtered['time'])
        mid_time_idx = n_times // 2
        mid_time = ds_filtered['time'].values[mid_time_idx]
        
        # Selecionar nível de pressão
        plevel_value = ds_filtered['plevels'].values[PLOT_PLEVEL_INDEX]
        
        print(f"{EMOJI_INFO} Tempo selecionado: {mid_time}")
        print(f"{EMOJI_INFO} Nível de pressão: {plevel_value:.0f} hPa (índice {PLOT_PLEVEL_INDEX})")
        
        # Extrair dados para o snapshot
        snapshot = ds_filtered.isel(time=mid_time_idx, plevels=PLOT_PLEVEL_INDEX)
        
        # Obter coordenadas
        if 'latitude' in ds_filtered.coords:
            lats = ds_filtered['latitude'].values
        else:
            lats = np.arange(ds_filtered.dims['latitude'])
        
        if 'longitude' in ds_filtered.coords:
            lons = ds_filtered['longitude'].values
        else:
            lons = np.arange(ds_filtered.dims['longitude'])
        
        # Criar meshgrid para plotagem
        if len(lats.shape) == 1 and len(lons.shape) == 1:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            lon_grid, lat_grid = lons, lats
        
        # Configurar figura com subplots
        n_vars = len(VARS_TO_KEEP)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig = plt.figure(figsize=(16, 5 * n_rows))
        
        # Títulos e configurações para cada variável
        var_configs = {
            'TT': {'title': 'Temperature', 'cmap': 'RdYlBu_r', 'unit': 'K'},
            'UU': {'title': 'Zonal Wind (U)', 'cmap': 'RdBu_r', 'unit': 'm/s'},
            'VV': {'title': 'Meridional Wind (V)', 'cmap': 'RdBu_r', 'unit': 'm/s'},
            'W': {'title': 'Vertical Velocity (Omega)', 'cmap': 'RdBu_r', 'unit': 'Pa/s'},
            'GHT': {'title': 'Geopotential Height', 'cmap': 'terrain', 'unit': 'm'}
        }
        
        for idx, var in enumerate(VARS_TO_KEEP):
            if var not in snapshot:
                continue
            
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection=ccrs.PlateCarree())
            
            # Dados da variável
            data = snapshot[var].values
            
            # Configuração da variável
            config = var_configs.get(var, {'title': var, 'cmap': 'viridis', 'unit': ''})
            
            # Plot
            if var == 'W':
                # Omega: usar apenas 70% do range para destacar movimentos ascendentes/descendentes
                vmax = np.nanmax(np.abs(data)) * 0.7
                im = ax.contourf(lon_grid, lat_grid, data, 
                               levels=20, cmap=config['cmap'], 
                               vmin=-vmax, vmax=vmax,
                               extend='both',
                               transform=ccrs.PlateCarree())
            elif var in ['UU', 'VV']:
                # Outras variáveis de vento - usar colormap divergente
                vmax = np.nanmax(np.abs(data))
                im = ax.contourf(lon_grid, lat_grid, data, 
                               levels=20, cmap=config['cmap'], 
                               vmin=-vmax, vmax=vmax,
                               transform=ccrs.PlateCarree())
            else:
                # Outras variáveis - usar colormap sequencial
                im = ax.contourf(lon_grid, lat_grid, data, 
                               levels=20, cmap=config['cmap'],
                               transform=ccrs.PlateCarree())
            
            # Adicionar coastlines e borders
            ax.coastlines(resolution='50m', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            
            # Gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                             color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
            # Título
            ax.set_title(f"{config['title']} [{config['unit']}]", fontsize=12, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label(config['unit'], fontsize=10)
        
        # Título geral
        time_str = str(mid_time).split('T')[0] if 'T' in str(mid_time) else str(mid_time)
        fig.suptitle(f'Snapshot: {time_str} - {plevel_value:.0f} hPa', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Salvar figura
        plot_filename = OUTFILE.replace('.nc', '_snapshot.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"{EMOJI_OK} Plot salvo: '{plot_filename}'")
        plt.close()
        
    except ImportError as e:
        print(f"{EMOJI_INFO} Aviso: não foi possível gerar plots. Instale matplotlib e cartopy: {e}")
    except Exception as e:
        print(f"{EMOJI_INFO} Aviso: erro ao gerar plots: {e}")

# Salvar o dataset processado
print(f"\n{EMOJI_INFO} Salvando arquivo processado: '{OUTFILE}'")
try:
    if USE_COMPRESSION:
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_filtered.data_vars}
        # Adicionar encoding para coordenadas também
        for coord in ds_filtered.coords:
            if coord not in encoding and coord in ['time', 'plevels', 'latitude', 'longitude']:
                encoding[coord] = {'_FillValue': None}
        ds_filtered.to_netcdf(OUTFILE, encoding=encoding)
    else:
        ds_filtered.to_netcdf(OUTFILE)
except Exception as e:
    print(f"{EMOJI_FAIL} Erro ao salvar arquivo: {e}")
    print(f"{EMOJI_INFO} Tentando salvar sem compressão...")
    ds_filtered.to_netcdf(OUTFILE)

print(f"{EMOJI_OK} Arquivo processado criado com sucesso: '{OUTFILE}'")

# Gerar arquivo namelist
print(f"\n{EMOJI_INFO} Gerando arquivo namelist...")
namelist_file = OUTFILE.replace('.nc', '_namelist')
try:
    # Mapeamento de nomes padrão para nomes descritivos
    standard_names = {
        'TT': 'Air Temperature',
        'UU': 'Eastward Wind Component',
        'VV': 'Northward Wind Component',
        'W': 'Omega Velocity',
        'GHT': 'Geopotential Height',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'time': 'Time',
        'plevels': 'Vertical Level'
    }
    
    # Obter unidades do dataset
    with open(namelist_file, 'w') as f:
        # Cabeçalho
        f.write('standard_name;Variable;Units\n')
        
        # Variáveis de dados
        for var in VARS_TO_KEEP:
            if var in ds_filtered:
                standard = standard_names.get(var, var)
                unit = ds_filtered[var].attrs.get('units', '')
                f.write(f'{standard};{var};{unit}\n')
        
        # Coordenadas - usar nomes reais das dimensões no arquivo processado
        coord_order = ['longitude', 'latitude', 'time', 'plevels']
        for coord in coord_order:
            if coord in ds_filtered.coords:
                standard = standard_names.get(coord, coord)
                unit = ds_filtered[coord].attrs.get('units', '')
                # Usar nomes das dimensões como estão no arquivo
                var_name = coord
                if unit == '':
                    if coord == 'longitude':
                        unit = 'degrees_east'
                    elif coord == 'latitude':
                        unit = 'degrees_north'
                    elif coord == 'time':
                        unit = 'datetime64'
                    elif coord == 'plevels':
                        unit = 'hPa'
                
                f.write(f'{standard};{var_name};{unit}\n')
    
    print(f"{EMOJI_OK} Namelist criado com sucesso: '{namelist_file}'")
    
except Exception as e:
    print(f"{EMOJI_INFO} Aviso: não foi possível criar namelist: {e}")

print(f"\n{EMOJI_PAPER} Resumo do dataset processado:")
print(ds_filtered)
