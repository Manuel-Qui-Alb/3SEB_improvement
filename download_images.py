import ee
import geemap
import pandas as pd
import gee.gee_objects as gee
import datetime as dt
import math

def equation_of_time_minutes(utc_dt: dt.datetime) -> float:
    """
    NOAA formulation (minutes). Works best with UTC datetime (timezone-aware or naive as UTC).
    """
    # Ensure we use UTC date/time
    if utc_dt.tzinfo is not None:
        utc_dt = utc_dt.astimezone(dt.timezone.utc).replace(tzinfo=None)

    year_start = dt.datetime(utc_dt.year, 1, 1)
    day_of_year = (utc_dt - year_start).days + 1
    # fractional hour in UTC
    frac_hour = utc_dt.hour + utc_dt.minute/60 + utc_dt.second/3600
    # fractional year (radians); NOAA variant
    gamma = 2*math.pi/365 * (day_of_year - 1 + (frac_hour - 12)/24)

    eot = (229.18 * (0.000075
                     + 0.001868*math.cos(gamma)
                     - 0.032077*math.sin(gamma)
                     - 0.014615*math.cos(2*gamma)
                     - 0.040849*math.sin(2*gamma)))
    return eot  # minutes (can be negative)


def utc_to_local_solar_time(utc_millis: int, longitude_deg: float):
    utc =  dt.datetime.fromtimestamp(utc_millis/1000, tz=dt.timezone.utc)
    lon_offset_min = 4.0 * longitude_deg
    lmst = utc + dt.timedelta(minutes=lon_offset_min)

    eot_min = equation_of_time_minutes(utc)
    ast = lmst + dt.timedelta(minutes=eot_min)
    return ast


def reduceRegion(image, band, vector, reducer='mean', scale=10, crs='EPSG:32610'):
    reducer = (image.select(band)
               .reduceRegion(reducer=reducer, geometry=vector, scale=scale, crs=crs)
               .set('system:time_start', image.get('system:time_start'))
               .set('MEAN_SOLAR_AZIMUTH_ANGLE', image.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
               .set('MEAN_SOLAR_ZENITH_ANGLE', image.get('MEAN_SOLAR_ZENITH_ANGLE'))
               .set('MEAN_INCIDENCE_ZENITH_ANGLE_B8', image.get('MEAN_INCIDENCE_ZENITH_ANGLE_B8'))
               )
    reducer = reducer.getInfo()
    return reducer


### Define study site and dates
# vector_BLS = ee.FeatureCollection('projects/saw-ucdavis/assets/BLS')
vector_RIP720 = ee.FeatureCollection('projects/saw-ucdavis/assets/RIP_720')

out_images = rf'C:\Users\mqalborn\Desktop\ET_3SEB\GRAPEX\Sentinel2'
out_metadata = rf'C:\Users\mqalborn\Desktop\ET_3SEB\GRAPEX\Sentinel2/METADATA.csv'
vector = vector_RIP720
farm = 'RIP720'
first_date = '2018-01-01'
last_date = '2018-05-31'
crs = 'EPSG:32610'

centroid = vector.geometry().centroid().getInfo()['coordinates']
lon = centroid[0]
lat = centroid[1]

# Load ImageCollection from Google Earth Engine
S2 = (gee.Sentinel2(farm=farm, aoi=vector)
      .filter_date(first_date, last_date)
      .cloud_mask())

S2 = S2.clip()
S2.percentage_pixel_free_clouds(band='B4', scale=10, crs=crs)
S2.filter_by_feature(filter='gte',
                     name='percentage_pixel_free_clouds',
                     value=99)

# Download images

geemap.ee_export_image_collection(
    S2.gee_image_collection.select('B.*'),
    scale=10,
    region=S2.aoi.geometry(),
    out_dir=out_images,
    crs=crs)

# Get AOT, WVP, azimuth, zenith and solar time
dates = S2.get_feature('date', unique=True)
data_stats = [reduceRegion(S2.get_image(i), band=['AOT', 'WVP'], vector=vector, scale=20, crs=crs)
                      for i in dates]
data_stats = pd.DataFrame(data_stats)
data_stats['overpass_solar_time'] = data_stats['system:time_start'].map(
    lambda x: utc_to_local_solar_time(x, longitude_deg=lon)
)
data_stats = data_stats[['overpass_solar_time', 'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_INCIDENCE_ZENITH_ANGLE_B8', 'AOT', 'WVP']]
data_stats.WVP = data_stats.WVP * 0.001
data_stats.AOT = data_stats.AOT * 0.001
data_stats.to_csv(out_metadata, index=False)