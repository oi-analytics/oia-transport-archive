
# -*- coding: utf-8 -*-
"""
Python script to intersect hazards and network line geometries
Created on Wed Jul 18 2018

@author: Raghav Pant
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import itertools
from shapely.geometry import Polygon
import rasterio
from openpyxl import load_workbook


from vtra.utils import *

def main():
	data_path = load_config()['paths']['data']
	output_path = load_config()['paths']['output']

	hazard_description_file = os.path.join(data_path,'Hazard_data','hazard_data_folder_data_info.xlsx')
	hazard_df = pd.read_excel(hazard_description_file,sheet_name ='file_contents')
	hazard_files = hazard_df['file_name'].values.tolist()

	book = load_workbook(hazard_description_file)
	excel_writer = pd.ExcelWriter(hazard_description_file,engine = 'openpyxl')
	excel_writer.book = book

	hazard_df['grid_x'] = 0
	hazard_df['grid_y'] = 0
	hazard_dir = os.path.join(data_path,'Hazard_data')
	for root, dirs, files in os.walk(hazard_dir):
		for file in files:
			if file.endswith(".tif") or file.endswith(".tiff"):
				hazard_shp = os.path.join(root,file)
				hazard_file = file.split(".")[0]
				print (hazard_file)
				if hazard_file in hazard_files:
					raster =  rasterio.open(hazard_shp)
					px,py = raster.res

					hazard_df.loc[hazard_df['file_name'] == hazard_file,'grid_x'] = px
					hazard_df.loc[hazard_df['file_name'] == hazard_file,'grid_y'] = py



	hazard_df.to_excel(excel_writer,'file_contents_2',index = False)
	excel_writer.save()


if __name__ == "__main__":
	main()
