import xarray as xr
import numpy as np
from metpy.units import units
from metpy.calc import vertical_velocity_pressure
from metpy.constants import g
import salem
# from wrf import interplevel

# INPUT_DATA = xr.open_mfdataset("data/*")
INPUT_FILE = "data/wrfout_d01_2021-06-26_05_00_00 (2)"
OUTPUT_FILE = "./Raoni_cowast_test.nc"

def main():

    input_data = salem.open_wrf_dataset(INPUT_FILE)
    bottom_top_coords = input_data['bottom_top'].values  # Extract the values from the bottom_top DataArray
    input_data = input_data.assign_coords({'bottom_top': bottom_top_coords})  # Assign these values as coordinates
    input_data = input_data.sortby('bottom_top', ascending=False)
    if 'xtime' in input_data.coords:
        input_data = input_data.drop('xtime')

    time = input_data.time
    print(f"times: {time.values}")

    # Levels to interpolate to
    plevs = [50.,   70.,
            100.,  125.,  150.,  175.,  200.,  225.,  250.,  300.,  350.,  400.,
            450.,  500.,  550.,  600.,  650.,  700.,  750.,  775.,  800.,  825.,
            850.,  875.,  900.,  925.,  950.,  975., 1000.] 
    print(f"levels to interpolate to: {plevs}")

    plevs = xr.DataArray(
        data=np.array(plevs * units.hPa),
        attrs=dict(long_name="isobaric_levels", units="hPa")).assign_coords(dim_0=plevs).rename({'dim_0':'level'}) 

    w_isob = input_data.salem.wrf_plevel('W', levels=plevs) * units('m/s')
    u_isob = input_data.salem.wrf_plevel('U', levels=plevs) * units('m/s')
    v_isob = input_data.salem.wrf_plevel('V', levels=plevs) * units('m/s')
    t_isob = input_data.salem.wrf_plevel('TK', levels=plevs)  * units('K')
    p_isob = input_data.salem.wrf_plevel('P', levels=plevs)
    qv_isob = input_data.salem.wrf_plevel('QVAPOR', levels=plevs) * units('kg/kg')

    # Variables that need to sum perturbation and base state
    hgt = (input_data['GEOPOTENTIAL'] / g * units('m')) + (input_data['PHB'] / g * units('m')) # geopotential height
    p = (input_data['P'] * units('Pa')).metpy.convert_units('hPa') + (input_data['PB'] * units('Pa')).metpy.convert_units('hPa') # pressure

    omega = vertical_velocity_pressure(w_isob, p_isob, t_isob, mixing_ratio=qv_isob)

    # Create dataset from variables
    output_data = t_isob.metpy.dequantify().to_dataset(name="tempk")
    output_data = output_data.assign(uwnd=u_isob.metpy.dequantify())
    output_data = output_data.assign(vwnd=v_isob.metpy.dequantify())
    output_data = output_data.assign(omega=omega.metpy.dequantify())
    output_data = output_data.assign(hgt=hgt.metpy.dequantify())

    output_data = output_data.assign_coords(level=plevs)

    output_data.attrs=dict(description="COWAST data for subtropical cyclone Raoni after 1) downgrading the spatial and temporal resolution \
    convert_mpas utility and 2) conversion to isobaric levels, using an script adapted from the MPAS-BR repository: https://github.com/pedrospeixoto/MPAS-BR.")

    output_data['tempk']['units']  = str(t_isob.metpy.units)
    output_data['tempk']['long_name']  = 'temperature'

    output_data['uwnd']['units']  = str(u_isob.metpy.units)
    output_data['uwnd']['long_name']  = 'wind meridional component'

    output_data['vwnd']['units']  = str(v_isob.metpy.units)
    output_data['vwnd']['long_name']  = 'wind zonal component'

    output_data['omega']['units']  = str(omega.metpy.units)
    output_data['omega']['long_name']  = 'vertical velocity in pressure levels'

    output_data['hgt']['units']  = str(hgt.metpy.units)
    output_data['hgt']['long_name']  = 'Geopotential  Height'

    encoding = {'time': {'units': 'hours since 2000-01-01', 'calendar': 'gregorian'}}
    output_data.to_netcdf(OUTPUT_FILE, encoding=encoding)
    print(OUTPUT_FILE + ' created!')

    ds = xr.open_dataset(OUTPUT_FILE)
    print(f"time of saved file: {ds.time.values}")

if __name__ == '__main__':
    main()
