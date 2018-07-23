"""Summarise hazard data

Read and map CVTS data
Author: Raghav Pant
Date: April 17, 2018
"""
import configparser
import csv
import glob
import os

import fiona
import fiona.crs
import rasterio



from sqlalchemy import create_engine
import subprocess as sp
import psycopg2
import osgeo.ogr as ogr

import pandas as pd
import copy

import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from geoalchemy2 import Geometry, WKTElement
from haversine import haversine

engine = create_engine('postgresql://raghavpant:data123@localhost:5432/vietnam_data')

root_dir = '/Users/raghavpant/Desktop/Vietnam_data/CVTS/20170801/02'
all_cvts = []
df_length = 0
for root, dirs, files in os.walk(root_dir):
	for fname in files:
		if fname.endswith(".csv"):
			fpath = os.path.join(root, fname)
			'''Load points and convert to geodataframe with coordinates'''    
			load_points = pd.read_csv(fpath,usecols = ['PlateNumber','Latitude','Longitude','Time'])
			load_points = load_points.sort_values(by=['Time'])
			load_points['distance check'] = [0]*len(load_points.index)
			# load_points['Time'] = pd.to_datetime(load_points['Time'],unit='s')
			# dist_diff = [0]
			# time_diff = [0]
			for index, row in load_points.iterrows():
				if index < len(load_points.index)-1:
					d_diff = 1000.0*haversine((row['Latitude'],row['Longitude']),(load_points.at[index+1,'Latitude'],load_points.at[index+1,'Longitude']))
					# dist_diff.append(d_diff)
					if d_diff > 100:
						load_points.at[index,'distance check'] = 1
						load_points.at[index+1,'distance check'] = 1

			
			# load_points['Time diff'] = (load_points['Time']-load_points['Time'].shift()).fillna(0)
			# load_points['Distance'] = dist_diff
			load_points = load_points[load_points['distance check'] == 1]
			all_cvts.append(load_points[['PlateNumber','Latitude','Longitude','Time']])

			# df_length += len(load_points.index)
			# if df_length > 100:
			# 	break
			del load_points
			# print ('done with file',fpath,df_length)


cvts_df = pd.concat(all_cvts)
cvts_df.to_csv('cvts_test.csv',index = False)
geometry = [Point(xy) for xy in zip(cvts_df.Latitude, cvts_df.Longitude)]
cvts_df = cvts_df.drop(['Latitude', 'Longitude'], axis=1)
crs = {'init': 'epsg:4326'}
points_gdp = gpd.GeoDataFrame(cvts_df, crs=crs, geometry=geometry)
points_gdp['geom'] = points_gdp['geometry'].apply(lambda x: WKTElement(x.wkt, srid=4326))

#drop the geometry column as it is now duplicative
points_gdp.drop('geometry', 1, inplace=True)
del cvts_df

print ('created geopandas dataframe from the points')
points_gdp.to_sql('cvts_test', engine, if_exists = 'replace', schema = 'public', index = True,dtype={'geom': Geometry('POINT', srid= 4326)})

del points_gdp
			
