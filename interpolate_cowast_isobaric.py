import xarray as xr
import numpy as np
from metpy.units import units
from metpy.calc import vertical_velocity_pressure
from metpy.constants import g
import salem

# Define input and output file paths
INPUT_FILE = "data/wrfout_d01_2021-06-26_05_00_00 (2)"
OUTPUT_FILE = "./Raoni_cowast_test.nc"

def preprocess_data(input_data):
    """Preprocess WRF dataset, adjust coordinates, and remove unnecessary variables."""
    # Extract and assign bottom_top coordinates
    bottom_top_coords = input_data['bottom_top'].values
    input_data = input_data.assign_coords({'bottom_top': bottom_top_coords})
    input_data = input_data.sortby('bottom_top', ascending=False)

    # Drop xtime if present
    if 'xtime' in input_data.coords:
        input_data = input_data.drop('xtime')

    return input_data

def get_pressure_levels():
    """Return an xarray DataArray of pressure levels to interpolate to."""
    plevs = [50., 70., 100., 125., 150., 175., 200., 225., 250., 300., 350., 
             400., 450., 500., 550., 600., 650., 700., 750., 775., 800., 825., 
             850., 875., 900., 925., 950., 975., 1000.]
    return xr.DataArray(
        data=np.array(plevs) * units.hPa,
        attrs=dict(long_name="isobaric_levels", units="hPa")
    ).assign_coords(dim_0=plevs).rename({'dim_0': 'level'})

def interpolate_variables(input_data, plevs):
    """Interpolate variables to the specified pressure levels."""
    w_isob = input_data.salem.wrf_plevel('W', levels=plevs) * units('m/s')
    u_isob = input_data.salem.wrf_plevel('U', levels=plevs) * units('m/s')
    v_isob = input_data.salem.wrf_plevel('V', levels=plevs) * units('m/s')
    t_isob = input_data.salem.wrf_plevel('TK', levels=plevs) * units('K')
    p_isob = input_data.salem.wrf_plevel('P', levels=plevs)
    qv_isob = input_data.salem.wrf_plevel('QVAPOR', levels=plevs) * units('kg/kg')

    return w_isob, u_isob, v_isob, t_isob, p_isob, qv_isob

def compute_derived_variables(input_data, w_isob, p_isob, t_isob, qv_isob):
    """Compute derived variables such as vertical velocity and geopotential height."""
    hgt = (input_data['GEOPOTENTIAL'] / g * units('m')) + (input_data['PHB'] / g * units('m'))
    omega = vertical_velocity_pressure(w_isob, p_isob, t_isob, mixing_ratio=qv_isob)
    
    return hgt, omega

def create_output_dataset(t_isob, u_isob, v_isob, omega, hgt, plevs):
    """Create an xarray dataset from interpolated and derived variables."""
    output_data = t_isob.metpy.dequantify().to_dataset(name="tempk")
    output_data = output_data.assign(uwnd=u_isob.metpy.dequantify())
    output_data = output_data.assign(vwnd=v_isob.metpy.dequantify())
    output_data = output_data.assign(omega=omega.metpy.dequantify())
    output_data = output_data.assign(hgt=hgt.metpy.dequantify())
    output_data = output_data.assign_coords(level=plevs)

    # Add metadata to the dataset
    output_data.attrs = dict(
        description="COWAST data for subtropical cyclone Raoni after 1) downgrading the spatial and temporal resolution "
                    "convert_mpas utility and 2) conversion to isobaric levels, using a script adapted from the "
                    "MPAS-BR repository: https://github.com/pedrospeixoto/MPAS-BR."
    )

    # Assign units and long names for variables
    output_data['tempk']['units'] = str(t_isob.metpy.units)
    output_data['tempk']['long_name'] = 'temperature'

    output_data['uwnd']['units'] = str(u_isob.metpy.units)
    output_data['uwnd']['long_name'] = 'wind meridional component'

    output_data['vwnd']['units'] = str(v_isob.metpy.units)
    output_data['vwnd']['long_name'] = 'wind zonal component'

    output_data['omega']['units'] = str(omega.metpy.units)
    output_data['omega']['long_name'] = 'vertical velocity in pressure levels'

    output_data['hgt']['units'] = str(hgt.metpy.units)
    output_data['hgt']['long_name'] = 'Geopotential Height'

    return output_data

def save_output(output_data, output_file):
    """Save the xarray dataset to a NetCDF file."""
    encoding = {'time': {'units': 'hours since 2000-01-01', 'calendar': 'gregorian'}}
    output_data.to_netcdf(output_file, encoding=encoding)
    print(f"{output_file} created!")

def main():
    # Load and preprocess data
    input_data = salem.open_wrf_dataset(INPUT_FILE)
    input_data = preprocess_data(input_data)
    
    # Get pressure levels
    plevs = get_pressure_levels()
    
    # Interpolate variables to isobaric levels
    w_isob, u_isob, v_isob, t_isob, p_isob, qv_isob = interpolate_variables(input_data, plevs)
    
    # Compute derived variables
    hgt, omega = compute_derived_variables(input_data, w_isob, p_isob, t_isob, qv_isob)
    
    # Create output dataset
    output_data = create_output_dataset(t_isob, u_isob, v_isob, omega, hgt, plevs)
    
    # Save output to NetCDF file
    save_output(output_data, OUTPUT_FILE)

    # Check the saved file
    ds = xr.open_dataset(OUTPUT_FILE)
    print(f"Time of saved file: {ds.time.values}")

if __name__ == '__main__':
    main()
