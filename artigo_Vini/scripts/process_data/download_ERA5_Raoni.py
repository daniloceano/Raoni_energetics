import cdsapi

# Criação do cliente CDSAPI
client = cdsapi.Client()

# Função para realizar o download de dados para um dia específico
def download_era5_data(year, month, day, target_filename):
    request = {
        'product_type': 'reanalysis',  # Tipo de produto
        'variable': [
            'u_component_of_wind',  # Componente u do vento
            'v_component_of_wind',  # Componente v do vento
            'vertical_velocity',    # Velocidade vertical (omega)
            'temperature',          # Temperatura
            'geopotential',         # Geopotencial
        ],
        'year': year,               # Ano
        'month': month,             # Mês
        'day': day,                 # Dia específico
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
        'pressure_level': [
            '10', '20', '30', '50', '70',
            '100', '125', '150', '175', '200', '225', '250', '300',
            '350', '400', '450', '500', '550', '600', '650', '700',
            '750', '775', '800', '825', '850', '875', '900', '925',
            '950', '975', '1000',
        ],
        'area': [
            -21, -41, -62, -37,  # Região de interesse [S, W, N, E]
        ],
        'format': 'netcdf',  # Formato do arquivo de saída
    }

    # Realizando a solicitação para baixar os dados
    client.retrieve('reanalysis-era5-pressure-levels', request, target_filename)
    print(f"Download do arquivo {target_filename} concluído!")

# Exemplo de dias como entrada
june_days = ['25', '26', '27', '28', '29', '30']
july_days = ['01', '02']

# Download para junho (2021-06-25 a 2021-06-30) - Requisição por dia
for day in june_days:
    target_filename = f'era5_data_202106{day}.nc'
    download_era5_data('2021', '06', day, target_filename)

# Download para julho (2021-07-01 a 2021-07-02) - Requisição por dia
for day in july_days:
    target_filename = f'era5_data_202107{day}.nc'
    download_era5_data('2021', '07', day, target_filename)
