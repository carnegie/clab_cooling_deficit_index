import xarray as xr
from historicalT import get_yearly_degree_time



def compute_degree_time_prediction(year, T_file):
    futureT = xr.open_dataset(T_file)['tas']
    futureT = futureT.sel(time=futureT['time.year'] == year)
    print(futureT)    

    yearly_future_deg_time = get_yearly_degree_time(futureT, None)
    
    print(yearly_future_deg_time)


futureT_file = '/groups/carnegie_poc/leiduan_memex/shared/tas_day_CanESM5_ssp585_r1i1p1f1_gn_20150101-21001231.nc'

compute_degree_time_prediction(2030, futureT_file)

