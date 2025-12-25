"""
🔍 Verificador de Integridade de Arquivos GFS GRIB2
===================================================

Este script verifica todos os arquivos GFS na pasta de dados e identifica
quais estão corrompidos ou com problemas de leitura.

Autor: Danilo
Data: 2025
"""

import os
import sys
from pathlib import Path
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Cores ANSI para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'

# Emojis
EMOJI_CHECK = "✅"
EMOJI_FAIL = "❌"
EMOJI_WARN = "⚠️"
EMOJI_FILE = "📄"
EMOJI_FOLDER = "📁"
EMOJI_SEARCH = "🔍"
EMOJI_ROCKET = "🚀"
EMOJI_CLOCK = "⏰"
EMOJI_SIZE = "📊"
EMOJI_CORRUPT = "💀"
EMOJI_OK = "✨"
EMOJI_DOWNLOAD = "📥"

# Configuração
GFS_DATA_DIR = '../../data/GFS'
FILE_PATTERN = 'gfs.0p25.*.grib2'

# Tamanho mínimo esperado para um arquivo GFS válido (em bytes)
# Arquivos GFS 0.25° completos geralmente têm pelo menos 150MB
MIN_FILE_SIZE = 150 * 1024 * 1024  # 150 MB


def print_header():
    """Imprime cabeçalho bonito."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {EMOJI_SEARCH} VERIFICADOR DE INTEGRIDADE - ARQUIVOS GFS GRIB2{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.ENDC}")
    print()


def print_section(title, emoji="📌"):
    """Imprime título de seção."""
    print()
    print(f"{Colors.BOLD}{Colors.YELLOW}  {emoji} {title}{Colors.ENDC}")
    print(f"{Colors.DIM}  {'─' * 60}{Colors.ENDC}")


def format_size(size_bytes):
    """Formata tamanho em bytes para formato legível."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def check_file_readable(filepath):
    """
    Verifica se o arquivo GRIB2 pode ser lido corretamente.
    Retorna (sucesso, mensagem, n_vars, n_levels)
    """
    try:
        import xarray as xr
        
        # Tentar abrir com cfgrib
        ds = xr.open_dataset(
            filepath,
            engine='cfgrib',
            filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 't'},
            backend_kwargs={'indexpath': ''}
        )
        
        # Verificar se tem dados
        n_times = len(ds['time']) if 'time' in ds.dims else 1
        n_levels = len(ds['isobaricInhPa']) if 'isobaricInhPa' in ds.dims else 0
        
        ds.close()
        
        if n_levels == 0:
            return False, "Sem níveis de pressão", 0, 0
        
        return True, "OK", 1, n_levels
        
    except Exception as e:
        error_msg = str(e)
        if 'No valid message' in error_msg:
            return False, "Arquivo vazio ou corrompido", 0, 0
        elif 'PrematureEndOfFile' in error_msg:
            return False, "Download incompleto", 0, 0
        elif 'WrongLength' in error_msg:
            return False, "Tamanho incorreto", 0, 0
        else:
            return False, f"Erro: {error_msg[:50]}", 0, 0


def extract_datetime_from_filename(filename):
    """Extrai data/hora do nome do arquivo GFS."""
    # gfs.0p25.2021062500.f000.grib2 -> 2021-06-25 00:00
    try:
        parts = filename.split('.')
        datetime_str = parts[2]  # 2021062500
        year = int(datetime_str[:4])
        month = int(datetime_str[4:6])
        day = int(datetime_str[6:8])
        hour = int(datetime_str[8:10])
        return datetime(year, month, day, hour)
    except:
        return None


def main():
    print_header()
    
    # Obter diretório
    script_dir = Path(__file__).parent
    data_dir = script_dir / GFS_DATA_DIR
    
    print(f"  {EMOJI_FOLDER} Diretório: {Colors.CYAN}{data_dir.resolve()}{Colors.ENDC}")
    
    # Encontrar arquivos
    file_pattern = str(data_dir / FILE_PATTERN)
    files = sorted(glob(file_pattern))
    
    if not files:
        print(f"\n  {EMOJI_FAIL} {Colors.RED}Nenhum arquivo encontrado!{Colors.ENDC}")
        sys.exit(1)
    
    print(f"  {EMOJI_FILE} Arquivos encontrados: {Colors.BOLD}{len(files)}{Colors.ENDC}")
    
    # Separar por verificações
    print_section("VERIFICAÇÃO DE TAMANHO", EMOJI_SIZE)
    
    files_info = []
    small_files = []
    
    for filepath in files:
        filename = Path(filepath).name
        size = os.path.getsize(filepath)
        dt = extract_datetime_from_filename(filename)
        
        info = {
            'path': filepath,
            'name': filename,
            'size': size,
            'datetime': dt,
            'size_ok': size >= MIN_FILE_SIZE,
            'readable': None,
            'read_msg': None
        }
        files_info.append(info)
        
        if not info['size_ok']:
            small_files.append(info)
    
    # Mostrar resumo de tamanhos
    sizes = [f['size'] for f in files_info]
    print(f"\n  {Colors.DIM}Tamanho mínimo esperado: {format_size(MIN_FILE_SIZE)}{Colors.ENDC}")
    print(f"  {Colors.DIM}Tamanho mínimo encontrado: {format_size(min(sizes))}{Colors.ENDC}")
    print(f"  {Colors.DIM}Tamanho máximo encontrado: {format_size(max(sizes))}{Colors.ENDC}")
    print(f"  {Colors.DIM}Tamanho médio: {format_size(sum(sizes) // len(sizes))}{Colors.ENDC}")
    
    if small_files:
        print(f"\n  {EMOJI_WARN} {Colors.YELLOW}Arquivos menores que o esperado: {len(small_files)}{Colors.ENDC}")
        for f in small_files:
            print(f"     {Colors.YELLOW}• {f['name']} ({format_size(f['size'])}){Colors.ENDC}")
    else:
        print(f"\n  {EMOJI_OK} {Colors.GREEN}Todos os arquivos têm tamanho adequado{Colors.ENDC}")
    
    # Verificar leitura
    print_section("VERIFICAÇÃO DE LEITURA (cfgrib)", EMOJI_SEARCH)
    print(f"\n  {Colors.DIM}Testando leitura de cada arquivo...{Colors.ENDC}\n")
    
    ok_files = []
    corrupted_files = []
    
    for i, info in enumerate(files_info):
        # Progress bar visual
        progress = (i + 1) / len(files_info)
        bar_width = 40
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        print(f"\r  [{bar}] {i+1}/{len(files_info)} - {info['name'][:30]}...", end='', flush=True)
        
        # Verificar arquivo
        readable, msg, n_vars, n_levels = check_file_readable(info['path'])
        info['readable'] = readable
        info['read_msg'] = msg
        info['n_levels'] = n_levels
        
        if readable:
            ok_files.append(info)
        else:
            corrupted_files.append(info)
    
    # Limpar linha de progresso
    print("\r" + " " * 80 + "\r", end='')
    
    # Mostrar resultados
    print_section("RESULTADO DA VERIFICAÇÃO", EMOJI_ROCKET)
    
    print(f"\n  {Colors.BOLD}Resumo:{Colors.ENDC}")
    print(f"  {'─' * 40}")
    print(f"  {EMOJI_CHECK} {Colors.GREEN}Arquivos OK:        {len(ok_files):3d}{Colors.ENDC}")
    print(f"  {EMOJI_CORRUPT} {Colors.RED}Arquivos corrompidos: {len(corrupted_files):3d}{Colors.ENDC}")
    print(f"  {'─' * 40}")
    print(f"  {EMOJI_FILE} Total:              {len(files_info):3d}")
    
    # Listar arquivos corrompidos
    if corrupted_files:
        print_section("ARQUIVOS CORROMPIDOS", EMOJI_CORRUPT)
        
        print(f"\n  {Colors.BOLD}{Colors.RED}{'Nome do Arquivo':<45} {'Tamanho':>10} {'Problema'}{Colors.ENDC}")
        print(f"  {Colors.RED}{'─' * 75}{Colors.ENDC}")
        
        for f in corrupted_files:
            size_str = format_size(f['size'])
            size_color = Colors.YELLOW if f['size'] < MIN_FILE_SIZE else Colors.ENDC
            print(f"  {Colors.RED}{f['name']:<45}{Colors.ENDC} {size_color}{size_str:>10}{Colors.ENDC} {Colors.DIM}{f['read_msg']}{Colors.ENDC}")
        
        # Gerar comandos de download
        print_section("COMANDOS PARA RE-DOWNLOAD", EMOJI_DOWNLOAD)
        
        print(f"\n  {Colors.CYAN}Os arquivos abaixo precisam ser baixados novamente:{Colors.ENDC}\n")
        
        # Base URL do GFS
        base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
        
        print(f"  {Colors.BOLD}Arquivos a baixar:{Colors.ENDC}")
        for f in corrupted_files:
            dt = f['datetime']
            if dt:
                date_str = dt.strftime('%Y%m%d')
                hour_str = f"{dt.hour:02d}"
                print(f"  {Colors.YELLOW}• {f['name']}{Colors.ENDC} ({dt.strftime('%Y-%m-%d %H:%M')})")
        
        print(f"\n  {Colors.BOLD}{Colors.CYAN}Script de download (copie e execute):{Colors.ENDC}\n")
        
        # Gerar script de download
        print(f"  {Colors.DIM}# Navegar para o diretório de dados{Colors.ENDC}")
        print(f"  cd {data_dir.resolve()}")
        print()
        
        for f in corrupted_files:
            dt = f['datetime']
            if dt:
                date_str = dt.strftime('%Y%m%d')
                hour_str = f"{dt.hour:02d}"
                # URL para dados históricos (AWS)
                aws_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date_str}/{hour_str}/atmos/gfs.t{hour_str}z.pgrb2.0p25.f000"
                
                print(f"  {Colors.DIM}# {f['name']}{Colors.ENDC}")
                print(f"  curl -o {f['name']} \"{aws_url}\"")
                print()
        
        # Alternativa com wget
        print(f"\n  {Colors.BOLD}{Colors.CYAN}Alternativa com wget:{Colors.ENDC}\n")
        for f in corrupted_files:
            dt = f['datetime']
            if dt:
                date_str = dt.strftime('%Y%m%d')
                hour_str = f"{dt.hour:02d}"
                aws_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date_str}/{hour_str}/atmos/gfs.t{hour_str}z.pgrb2.0p25.f000"
                print(f"  wget -O {f['name']} \"{aws_url}\"")
        
        # Salvar lista em arquivo
        list_file = data_dir / 'corrupted_files.txt'
        with open(list_file, 'w') as f:
            f.write("# Arquivos GFS corrompidos\n")
            f.write(f"# Verificado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for info in corrupted_files:
                f.write(f"{info['name']}\t{format_size(info['size'])}\t{info['read_msg']}\n")
        
        print(f"\n  {EMOJI_FILE} Lista salva em: {Colors.CYAN}{list_file}{Colors.ENDC}")
        
    else:
        print(f"\n  {EMOJI_OK} {Colors.GREEN}{Colors.BOLD}Todos os arquivos estão íntegros!{Colors.ENDC}")
    
    # Listar arquivos OK (agrupados por data)
    if ok_files:
        print_section("ARQUIVOS VÁLIDOS", EMOJI_CHECK)
        
        # Agrupar por data
        dates = {}
        for f in ok_files:
            dt = f['datetime']
            if dt:
                date_key = dt.strftime('%Y-%m-%d')
                if date_key not in dates:
                    dates[date_key] = []
                dates[date_key].append(f)
        
        for date_key in sorted(dates.keys()):
            files_in_date = dates[date_key]
            hours = ', '.join([f['datetime'].strftime('%H') for f in sorted(files_in_date, key=lambda x: x['datetime'])])
            print(f"  {Colors.GREEN}• {date_key}{Colors.ENDC}: {Colors.DIM}{hours}Z{Colors.ENDC}")
    
    # Rodapé
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.ENDC}")
    print(f"  {EMOJI_CLOCK} Verificação concluída em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.ENDC}")
    print()
    
    # Retornar código de saída
    return 1 if corrupted_files else 0


if __name__ == '__main__':
    sys.exit(main())
