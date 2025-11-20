import xarray as xr
import sys


INFILE = 'WRF_sacoplamento-RAONI-6h_INTRP-Regular.nc'
OUTFILE = 'WRF_sacoplamento-RAONI-6h_INTRP-Regular_filtered.nc'

# Emojis imersivos para feedback
EMOJI_START = "🚀"
EMOJI_OK = "✅"
EMOJI_FAIL = "❌"
EMOJI_INFO = "🔎"
EMOJI_PAPER = "📄"


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


print(f"{EMOJI_START} Iniciando pré-processamento: abrindo '{INFILE}'")

# Abrir o dataset
ds = xr.open_dataset(INFILE)

# Selecionar apenas as variáveis desejadas
vars_to_keep = ['TT', 'UU', 'VV', 'GHT', 'W']

# Verificar se todas as variáveis existem no dataset
available = set(ds.variables.keys())
missing = [v for v in vars_to_keep if v not in available]
if missing:
	print(f"\n{EMOJI_FAIL} Variável(s) não encontrada(s): {', '.join(missing)}")
	print_available_variables(ds)
	print(f"\n{EMOJI_FAIL} Abortando execução. Por favor, ajuste 'vars_to_keep' com base nas variáveis disponíveis.")
	sys.exit(1)

# Se chegou aqui, todas as variáveis existem; selecionar
ds_filtered = ds[vars_to_keep]

# Adicionar a variável de tempo se existir
if 'time' in ds.coords:
	try:
		ds_filtered = ds_filtered.set_coords('time')
		ds_filtered = ds_filtered.set_index(Time='time')
	except Exception:
		# Não bloquear a execução só por conta do index se algo for diferente
		print(f"{EMOJI_INFO} Aviso: não foi possível setar 'time' como coordenada/index. Continuando...")

# Remover variáveis que podem não existir (safety)
for v in ['XTIME', 'Times']:
	if v in ds_filtered:
		ds_filtered = ds_filtered.drop_vars(v)

# Salvar com compressão (opcional)
encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_filtered.data_vars}
ds_filtered.to_netcdf(OUTFILE, encoding=encoding, mode='w')

print(f"{EMOJI_OK} Arquivo filtrado criado com sucesso: '{OUTFILE}'")
print(ds_filtered)