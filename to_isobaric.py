import xarray as xr
import numpy as np
from metpy.units import units
from metpy.calc import vertical_velocity_pressure
from metpy.constants import g
from wrf import interplevel

input_data = xr.open_mfdataset("data_vCPAM/coawst/*.nc")
output_name = "data_vCPAM/Raoni_cowast.nc"

latitude_name, longitude_name = 'latitude', 'longitude'

bottom_top_coords = input_data['bottom_top'].values  # Extract the values from the bottom_top DataArray
input_data = input_data.assign_coords({'bottom_top': bottom_top_coords})  # Assign these values as coordinates
input_data = input_data.sortby('bottom_top', ascending=False)
input_data = input_data.drop('xtime')

# # Pre-process spatial coordinates
# input_data['south_north'] = input_data[latitude_name][:,0]
# input_data['west_east'] = input_data[longitude_name][0,:]
# input_data = input_data.rename({'south_north': 'latitude', 'west_east': 'longitude'})

time = input_data.time
print(f"times: {time.values}")

# Levels to interpolate to
plevs = [50.,   70.,
        100.,  125.,  150.,  175.,  200.,  225.,  250.,  300.,  350.,  400.,
        450.,  500.,  550.,  600.,  650.,  700.,  750.,  775.,  800.,  825.,
        850.,  875.,  900.,  925.,  950.,  975., 1000.] 
print(f"levels to interpolate to: {plevs}")

plevs = xr.DataArray(
    data=np.array(plevs* units.hPa),
    attrs=dict(long_name="isobaric_levels", units="hPa")).assign_coords(dim_0=plevs).rename({'dim_0':'level'}) 

# Open variables
print('opening variables...')
v = input_data['V'] * units('m/s') # v-component of wind
u = input_data['U'] * units('m/s') # u-component of wind
t = input_data['TK'] * units('K') # temperature
w = input_data['W'] * units('m/s') # vertical velocity
qv = input_data['QVAPOR'] * units('kg/kg')

# Variables that need to sum pertubation and base state
hgt = (input_data['PH'] / g * units('m')) + (input_data['PHB'] / g * units('m')) # geopotential height
p = (input_data['P'] * units('Pa')).metpy.convert_units('hPa') + (input_data['PB'] * units('Pa')).metpy.convert_units('hPa') # pressure

u_isob = interplevel(u, p, plevs).assign_coords(time=time) * u.metpy.units
v_isob = interplevel(v, p, plevs).assign_coords(time=time) * v.metpy.units
t_isob = interplevel(t, p, plevs).assign_coords(time=time) * t.metpy.units
w_isob = interplevel(w, p, plevs).assign_coords(time=time) * w.metpy.units
p_isob = interplevel(p, p, plevs).assign_coords(time=time) * p.metpy.units
qv_isob = interplevel(qv, p, plevs).assign_coords(time=time) * qv.metpy.units
hgt_isob = interplevel(hgt, p, plevs).assign_coords(time=time) * hgt.metpy.units
omega = vertical_velocity_pressure(w_isob, p_isob, t_isob, mixing_ratio=qv_isob)

# Create dataset from variables
output_data = t_isob.metpy.dequantify().to_dataset(name="tempk")
output_data = output_data.assign(uwnd=u_isob.metpy.dequantify())
output_data = output_data.assign(vwnd=v_isob.metpy.dequantify())
output_data = output_data.assign(omega=omega.metpy.dequantify())
output_data = output_data.assign(hgt=hgt_isob.metpy.dequantify())

output_data = output_data.assign_coords(level=plevs)

output_data.attrs=dict(description="COWAST data for subtropical cyclone Raoni after 1) downgrading the spatial and temporal reoslution\
convert_mpas utility and 2) conversion to isobaric levels, using an script adapted from the\
MPAS-BR repository: https://github.com/pedrospeixoto/MPAS-BR.")


output_data['tempk']['units']  = str(t_isob.metpy.units)
output_data['tempk']['long_name']  = 'temperature'

output_data['uwnd']['units']  = str(u_isob.metpy.units)
output_data['uwnd']['long_name']  = 'wind meridional component'

output_data['vwnd']['units']  = str(v_isob.metpy.units)
output_data['vwnd']['long_name']  = 'wind zonal component'

output_data['omega']['units']  = str(omega.metpy.units)
output_data['omega']['long_name']  = 'vertical velocity in pressure levels'

output_data['hgt']['units']  = str(hgt_isob.metpy.units)
output_data['hgt']['long_name']  = 'Geopotential  Height'

encoding = {'time': {'units': 'hours since 2000-01-01', 'calendar': 'gregorian'}}
output_data.to_netcdf(output_name, encoding=encoding)
print(output_name+' created!')

ds = xr.open_dataset(output_name)
print(f"time of saved file: {ds.time.values}")