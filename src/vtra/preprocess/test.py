# # -*- coding: utf-8 -*-
# """
# This is a rubbish script
# It is written to test different things
# """

# import os
# import subprocess
# import json
# import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# from vtra.utils import load_config
# from vtra.transport_network_creation import province_shapefile_to_network

# import fiona
# import fiona.crs
# import rasterio
# import geopandas as gpd
# import numpy as np
# import pandas as pd
# import igraph as ig
# from shapely.geometry import Point

# def raster_rewrite(in_raster,out_raster,nodata):
# 	with rasterio.open(in_raster) as dataset:
# 		data_array = dataset.read()
# 		data_array[np.where(np.isnan(data_array))] = nodata

# 		with rasterio.open(out_raster, 'w', driver='GTIff',
# 					height=data_array.shape[1],    # numpy of rows
# 					width=data_array.shape[2],     # number of columns
# 					count=dataset.count,                        # number of bands
# 					dtype=data_array.dtype,  # this must match the dtype of our array
# 					crs=dataset.crs,
# 					transform=dataset.transform) as out_data:
# 			out_data.write(data_array)  # optional second parameter is the band number to write to
# 			out_data.nodata = -1  # set the raster's nodata value


# 	os.remove(in_raster)
# 	os.rename(out_raster,in_raster)

# def raster_projections_and_databands(file_path):
# 	with rasterio.open(file_path) as dataset:
# 		counts = dataset.count
# 		if dataset.crs:
# 			crs = dataset.crs.to_string()
# 		else:
# 			crs = 'invalid/unknown'
# 		# bands = dataset.meta
# 		data_array = dataset.read()
# 		# print (data_array)
# 		if dataset.count > 1:
# 			data_list = []
# 			for i in range(0,dataset.count):
# 				data_list.append(data_array[i].reshape(dataset.height*dataset.width).tolist())
# 			data_vals = list(set(list(zip(*data_list))))
# 		else:
# 			data_vals = list(set(data_array.reshape(dataset.count*dataset.height*dataset.width).tolist()))
# 			if all(isinstance(x, int) for x in data_vals) is False:
# 				data_vals = []

# 		# print (list(set(resphaped_array.tolist())))
# 	return counts,crs, data_vals

# def convert_geotiff_to_vector_with_threshold(from_threshold,to_threshold, infile, infile_epsg,tmpfile_1, tmpfile_2, outfile):
# 	"""Threshold raster, convert to polygons, assign crs
# 	"""
# 	subprocess.run([
# 		"gdal_edit.py",
# 		'-a_srs', 'EPSG:{}'.format(infile_epsg),
# 		infile
# 	])

# 	args = [
# 		"gdal_calc.py",
# 		'-A', infile,
# 		'--outfile={}'.format(tmpfile_1),
# 		'--calc=logical_and(A>={0}, A<{1})'.format(from_threshold,to_threshold),
# 		'--type=Byte', '--NoDataValue=0',
# 		'--co=SPARSE_OK=YES',
# 		'--co=NBITS=1',
# 		'--quiet',
# 		'--co=COMPRESS=LZW'
# 	]
# 	subprocess.run(args)

# 	subprocess.run([
# 		"gdal_polygonize.py",
# 		tmpfile_1,
# 		'-q',
# 		'-f', 'ESRI Shapefile',
# 		tmpfile_2
# 	])

# 	subprocess.run([
# 		"ogr2ogr",
# 		'-a_srs', 'EPSG:{}'.format(infile_epsg),
# 		'-t_srs', 'EPSG:4326',
# 		outfile,
# 		tmpfile_2
# 	])

# 	subprocess.run(["rm", tmpfile_1])
# 	subprocess.run(["rm", tmpfile_2])
# 	subprocess.run(["rm", tmpfile_2.replace('shp', 'shx')])
# 	subprocess.run(["rm", tmpfile_2.replace('shp', 'dbf')])
# 	subprocess.run(["rm", tmpfile_2.replace('shp', 'prj')])

# def convert_geotiff_to_vector_with_multibands(band_colors, infile, infile_epsg,tmpfile_1, tmpfile_2, outfile):
# 	"""Threshold raster, convert to polygons, assign crs
# 	"""
# 	subprocess.run([
# 		"gdal_edit.py",
# 		'-a_srs', 'EPSG:{}'.format(infile_epsg),
# 		infile
# 	])

# 	args = [
# 		"gdal_calc.py",
# 		'-A', infile,
# 		'--A_band=1',
# 		'-B', infile,
# 		'--B_band=2',
# 		'-C', infile,
# 		'--C_band=3',
# 		'--outfile={}'.format(tmpfile_1),
# 		'--calc=logical_and(A=={0}, B=={1},C=={2})'.format(band_colors[0],band_colors[1],band_colors[2]),
# 		'--type=Byte', '--NoDataValue=0',
# 		'--co=SPARSE_OK=YES',
# 		'--co=NBITS=1',
# 		'--quiet',
# 		'--co=COMPRESS=LZW'
# 	]
# 	subprocess.run(args)

# 	subprocess.run([
# 		"gdal_polygonize.py",
# 		tmpfile_1,
# 		'-q',
# 		'-f', 'ESRI Shapefile',
# 		tmpfile_2
# 	])

# 	subprocess.run([
# 		"ogr2ogr",
# 		'-a_srs', 'EPSG:{}'.format(infile_epsg),
# 		'-t_srs', 'EPSG:4326',
# 		outfile,
# 		tmpfile_2
# 	])

# 	subprocess.run(["rm", tmpfile_1])
# 	subprocess.run(["rm", tmpfile_2])
# 	subprocess.run(["rm", tmpfile_2.replace('shp', 'shx')])
# 	subprocess.run(["rm", tmpfile_2.replace('shp', 'dbf')])
# 	subprocess.run(["rm", tmpfile_2.replace('shp', 'prj')])

# def convert(threshold, infile, tmpfile_1, outfile):
# 	"""Threshold raster, convert to polygons
# 	"""
# 	args = [
# 		"gdal_calc.py",
# 		'-A', infile,
# 		'--outfile={}'.format(tmpfile_1),
# 		'--calc=logical_and(A>={}, A<999)'.format(threshold),
# 		'--type=Byte', '--NoDataValue=0',
# 		'--co=SPARSE_OK=YES',
# 		'--co=NBITS=1',
# 		'--co=COMPRESS=LZW'
# 	]
# 	subprocess.run(args)

# 	subprocess.run([
# 		"gdal_polygonize.py",
# 		tmpfile_1,
# 		'-q',
# 		'-f', 'ESRI Shapefile',
# 		outfile
# 	])

# def main():
# 	# data_path = load_config()['paths']['data']
# 	# data_file = os.path.join(data_path,'Hazard_data','15_northern_provinces_flashfloods_susceptibility','FFSZ_2015.tif')
# 	# out_file = os.path.join(data_path,'Hazard_data','15_northern_provinces_flashfloods_susceptibility','test.tif')
# 	# raster_rewrite(data_file,out_file,0)

# 	data_path,calc_path,output_path = load_config()['paths']['data'],load_config()['paths']['calc'],load_config()['paths']['output']

# 	rd_prop_file = os.path.join(data_path,'Roads','road_properties','road_properties.xlsx')
# 	# print (rd_prop_file)
# 	# provinces to consider
# 	province_list = ['Lao Cai','Thanh Hoa','Binh Dinh']
# 	# province_list = ['Lao Cai']
# 	road_terrain_list = ['mountain','flat','flat']

# 	crop_data_path = os.path.join(data_path,'Agriculture_crops','crop_data')
# 	crop_names = ['rice','cash','cass','teas','maiz','rubb','swpo','acof','rcof','pepp']
# 	crop_month_file = os.path.join(data_path,'rice_atlas_vietnam','rice_production.shp')
# 	crop_month_fields = ['P_Jan','P_Feb','P_Mar','P_Apr','P_May','P_Jun','P_Jul','P_Aug','P_Sep','P_Oct','P_Nov','P_Dec']

# 	for prn in range(len(province_list)):
# 		province = province_list[prn]
# 		# set all paths for all input files we are going to use
# 		province_name = province.replace(' ','').lower()
# 		edges_in = os.path.join(data_path,'Roads','{}_roads'.format(province_name),'vietbando_{}_edges.shp'.format(province_name))
# 		# find the crop production months for the province
# 		crop_prod_months = gpd.read_file(crop_month_file)
# 		crop_prod_months = crop_prod_months.loc[crop_prod_months.SUB_REGION == province]
# 		crop_prod_months = crop_prod_months[crop_month_fields].values.tolist()
# 		# print (crop_prod_months)
# 		crop_prod_months = np.array(crop_prod_months[0])/sum(crop_prod_months[0])
# 		print (province,crop_prod_months)
# 		crop_prod_months = crop_prod_months[crop_prod_months > 0]
# 		print (province,crop_prod_months.tolist())

# 	# 	# load network

# 	# 	# G = province_shapefile_to_network(edges_in,path_width_table)
# 	# 	G = province_shapefile_to_network(edges_in,road_terrain_list[prn],rd_prop_file)
# 	# 	# edgs.to_csv(os.path.join(output_path,'{}_network.csv'.format(province_name)),index = False)
# 	# 	# G = add_igraph_time_costs_province_roads(G,0.019)

# 	# 	# nodes_name = np.asarray([x['name'] for x in G.vs])
# 	# 	# nodes_index = np.asarray([x.index for x in G.vs])
# 	# 	# node_dict = dict(zip(nodes_name,nodes_index))

# 	# for file in os.listdir(crop_data_path):
# 	# 	if file.endswith(".tif") and 'spam_p' in file.lower().strip():
# 	# 		fpath = os.path.join(crop_data_path, file)
# 	# 		crop_name = [cr for cr in crop_names if cr in file.lower().strip()][0]
# 	# 		outCSVName = os.path.join(output_path,'crop_flows','crop_concentrations.csv')
# 	# 		subprocess.run(["gdal2xyz.py",'-csv', fpath,outCSVName])

# 	# 		'''Load points and convert to geodataframe with coordinates'''
# 	# 		load_points = pd.read_csv(outCSVName,header=None,names=['x','y','tons'],index_col=None)
# 	# 		load_points = load_points[load_points['tons'] > 0]
# 	# 		crop_tot = load_points['tons'].sum(axis = 0)
# 	# 		print (crop_name,crop_tot)
# 	# band_nums, crs, unique_data_values = raster_projections_and_databands(data_file)
# 	# print (band_nums,crs, unique_data_values)

# 	# root_dir = os.path.join(data_path,'Hazard_data')
# 	# for root, dirs, files in os.walk(root_dir):
# 	# 	for file in files:
# 	# 		if file.endswith(".tif") or file.endswith(".tiff"):
# 	# 			# fpath = os.path.join(root, fname)
# 	# 			band_nums, crs, unique_data_values = raster_projections_and_databands(os.path.join(root, file))
# 	# 			print (file,crs, unique_data_values)
# 	# 			if 'WGS84' in crs:
# 	# 				s_crs = 4326
# 	# 			elif crs != 'invalid/unknown':
# 	# 				crs_split = crs.split(':')
# 	# 				s_crs = [int(c) for c in crs_split if c.isdigit() is True][0]
# 	# 			else:
# 	# 				s_crs = 32648

# 	# 			if not unique_data_values:
# 	# 				# threshold based datasets
# 	# 				thresholds = [1,2,3,4,999]
# 	# 				for t in range(len(thresholds)-1):
# 	# 					thr_1 = thresholds[t]
# 	# 					thr_2 = thresholds[t+1]
# 	# 					in_file = os.path.join(root,file)
# 	# 					tmp_1 = os.path.join(root,file.split(".tif")[0] + '_mask.tiff')
# 	# 					tmp_2 = os.path.join(root,file.split(".tif")[0] + '_mask.shp')
# 	# 					out_file = os.path.join(root,file.split(".tif")[0] + '_{0}m-{1}m_threshold.shp'.format(thr_1,thr_2))
# 	# 					convert_geotiff_to_vector_with_threshold(thr_1,thr_2,in_file,s_crs,tmp_1, tmp_2, out_file)
# 	# 			elif band_nums == 1:
# 	# 				# code value based dataset
# 	# 				code_vals = [4,5]
# 	# 				for c in code_vals:
# 	# 					in_file = os.path.join(root,file)
# 	# 					tmp_1 = os.path.join(root,file.split(".tif")[0] + '_mask.tiff')
# 	# 					tmp_2 = os.path.join(root,file.split(".tif")[0] + '_mask.shp')
# 	# 					out_file = os.path.join(root,file.split(".tif")[0] + '_{}_band.shp'.format(c))
# 	# 					convert_geotiff_to_vector_with_threshold(c,c+1,in_file,s_crs,tmp_1, tmp_2, out_file)
# 	# 			elif band_nums == 3:
# 	# 				# multi-band color datasets
# 	# 				for dv in unique_data_values:
# 	# 					if dv in [(255,190,190),(245,0,0),(255,0,0)]:
# 	# 						thr = 5
# 	# 						bc = dv
# 	# 						in_file = os.path.join(root,file)
# 	# 						tmp_1 = os.path.join(root,file.split(".tif")[0] + '_mask.tiff')
# 	# 						tmp_2 = os.path.join(root,file.split(".tif")[0] + '_mask.shp')
# 	# 						out_file = os.path.join(root,file.split(".tif")[0] + '_{}_band.shp'.format(thr))
# 	# 						convert_geotiff_to_vector_with_multibands(bc,in_file,s_crs,tmp_1, tmp_2, out_file)
# 	# 					elif dv in  [(255,170,0),(255,128,0)]:
# 	# 						thr = 4
# 	# 						bc = dv

# 	# 						in_file = os.path.join(root,file)
# 	# 						tmp_1 = os.path.join(root,file.split(".tif")[0] + '_mask.tiff')
# 	# 						tmp_2 = os.path.join(root,file.split(".tif")[0] + '_mask.shp')
# 	# 						out_file = os.path.join(root,file.split(".tif")[0] + '_{}_band.shp'.format(thr))
# 	# 						convert_geotiff_to_vector_with_multibands(bc,in_file,s_crs,tmp_1, tmp_2, out_file)





# if __name__ == "__main__":
# 	main()



# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:41:39 2018

@author: elcok
"""

import geopandas as gpd
import pandas as pd
import os
import igraph as ig
import numpy as np
import sys
import subprocess
from shapely.geometry import Point, LineString
import itertools
import operator
import ast
import math
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import shapely.geometry
import shapely.ops
from shapely.geometry import Polygon

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from vtra.utils import *
from vtra.transport_network_creation import *

def assign_value_in_area_proportions(poly_1_gpd,poly_2_gpd,poly_attribute):
	poly_1_sindex = poly_1_gpd.sindex
	for p_2_index, polys_2 in poly_2_gpd.iterrows():
		poly2_attr= 0
		intersected_polys = poly_1_gpd.iloc[list(poly_1_sindex.intersection(polys_2.geometry.bounds))]
		for p_1_index, polys_1 in intersected_polys.iterrows():
			if (polys_2['geometry'].intersects(polys_1['geometry']) is True) and (polys_1.geometry.is_valid is True) and (polys_2.geometry.is_valid is True):
				poly2_attr += polys_1[poly_attribute]*polys_2['geometry'].intersection(polys_1['geometry']).area/polys_1['geometry'].area

		poly_2_gpd.loc[p_2_index,poly_attribute] = poly2_attr

	return poly_2_gpd

def get_nearest_node_on_line(x,lines_df,line_id_column):
	# line_geom = get_nearest_node(x,sindex_lines,lines_df,line_geometry_column)
	# line_geom = lines_df.loc[list(lines_df.geometry.nearest(x))][line_geometry_column].values[0]
	min_dist = 1e10
	line_geom = 0
	line_id = ''
	for l_iter,l_val in lines_df.iterrows():
		p_dist = x.distance(l_val.geometry)
		if p_dist <= min_dist:
			min_dist = p_dist
			line_geom = l_val.geometry
			line_id = l_val[line_id_column]
	# line_geom = lines_df.loc[list(sindex_lines.nearest(x))][line_geometry_column].values[0]
	return line_id,line_geom.interpolate(line_geom.project(x))


def voronoi_finite_polygons_2d(vor, radius=None):
	"""
	Copy-pasted from: https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python

	Reconstruct infinite voronoi regions in a 2D diagram to finite
	regions.
	Parameters
	----------
	vor : Voronoi
		Input diagram
	radius : float, optional
		Distance to 'points at infinity'.
	Returns
	-------
	regions : list of tuples
		Indices of vertices in each revised Voronoi regions.
	vertices : list of tuples
		Coordinates for revised Voronoi vertices. Same as coordinates
		of input vertices, with 'points at infinity' appended to the
		end.
	"""

	if vor.points.shape[1] != 2:
		raise ValueError("Requires 2D input")

	new_regions = []
	new_vertices = vor.vertices.tolist()

	center = vor.points.mean(axis=0)
	if radius is None:
		radius = vor.points.ptp().max()*2

	# Construct a map containing all ridges for a given point
	all_ridges = {}
	for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
		all_ridges.setdefault(p1, []).append((p2, v1, v2))
		all_ridges.setdefault(p2, []).append((p1, v1, v2))

	# Reconstruct infinite regions
	for p1, region in enumerate(vor.point_region):
		vertices = vor.regions[region]

		if all(v >= 0 for v in vertices):
			# finite region
			new_regions.append(vertices)
			continue

		# reconstruct a non-finite region
		ridges = all_ridges[p1]
		new_region = [v for v in vertices if v >= 0]

		for p2, v1, v2 in ridges:
			if v2 < 0:
				v1, v2 = v2, v1
			if v1 >= 0:
				# finite ridge: already in the region
				continue

			# Compute the missing endpoint of an infinite ridge

			t = vor.points[p2] - vor.points[p1] # tangent
			t /= np.linalg.norm(t)
			n = np.array([-t[1], t[0]])  # normal

			midpoint = vor.points[[p1, p2]].mean(axis=0)
			direction = np.sign(np.dot(midpoint - center, n)) * n
			far_point = vor.vertices[v2] + direction * radius

			new_region.append(len(new_vertices))
			new_vertices.append(far_point.tolist())

		# sort region counterclockwise
		vs = np.asarray([new_vertices[v] for v in new_region])
		c = vs.mean(axis=0)
		angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
		new_region = np.array(new_region)[np.argsort(angles)]

		# finish
		new_regions.append(new_region.tolist())

	return new_regions, np.asarray(new_vertices)

def extract_nodes_within_gdf(x,input_nodes,column_name):
	return input_nodes.loc[list(input_nodes.geometry.within(x.geometry))][column_name].values[0]

def netrev_edges(region_name,start_points,end_points,graph,save_edges = True,output_path ='',excel_writer =''):
	"""
	Assign net revenue to roads assets in Vietnam

	Inputs are:
	start_points - GeoDataFrame of start points for shortest path analysis.
	end_points - GeoDataFrame of potential end points for shorest path analysis.
	G - iGraph network of the province.
	save_edges -

	Outputs are:
	Shapefile with all edges and the total net reveneu transferred along each edge
	GeoDataFrame of total net revenue transferred along each edge
	"""
	save_paths = []
	path_index = 0
	for iter_,place in start_points.iterrows():
		try:
			closest_center = end_points.loc[end_points['OBJECTID']
			== place['NEAREST_C_CENTER']]['NEAREST_G_NODE'].values[0]

			pos0_i = graph.vs[node_dict[place['NEAREST_G_NODE']]]
			pos1_i = graph.vs[node_dict[closest_center]]

			if pos0_i != pos1_i:
				path = graph.get_shortest_paths(pos0_i,pos1_i,weights='min_cost',output="epath")
				get_od_pair = (place['NEAREST_G_NODE'],closest_center)
				get_path = [graph.es[n]['edge_id'] for n in path][0]
				get_dist = sum([graph.es[n]['length'] for n in path][0])
				get_time = sum([graph.es[n]['min_time'] for n in path][0])
				get_travel_cost = sum([graph.es[n]['min_cost'] for n in path][0])
				path_index += 1
				save_paths.append(('path_{}'.format(path_index),get_od_pair,get_path,place['netrev'],get_travel_cost,get_dist,get_time))
		except:
			print(iter_)


	save_paths_df = pd.DataFrame(save_paths,columns = ['path_index','od_nodes','edge_path','netrev','travel_cost','distance','time'])
	save_paths_df.to_excel(excel_writer,province_name,index = False)
	excel_writer.save()
	del save_paths_df

	all_edges = [x['edge_id'] for x in graph.es]
	all_edges_geom = [x['geometry'] for x in graph.es]

	gdf_edges = gpd.GeoDataFrame(pd.DataFrame([all_edges,all_edges_geom]).T,crs='epsg:4326')
	gdf_edges.columns = ['edge_id','geometry']

	gdf_edges['netrev'] = 0
	for path in save_paths:
		gdf_edges.loc[gdf_edges['edge_id'].isin(path[2]),'netrev'] += path[3]

	if save_edges == True:
		gdf_edges.to_file(os.path.join(output_path,'weighted_edges_district_center_flows_{}.shp'.format(region_name)))
	return gdf_edges

def network_od_paths_assembly(points_dataframe,node_dict,graph,vehicle_wt,region_name,save_edges = True,output_path ='',excel_writer =''):
	"""
	Assign net revenue to roads assets in Vietnam

	Inputs are:
	start_points - GeoDataFrame of start points for shortest path analysis.
	end_points - GeoDataFrame of potential end points for shorest path analysis.
	G - iGraph network of the province.
	save_edges -

	Outputs are:
	Shapefile with all edges and the total net reveneu transferred along each edge
	GeoDataFrame of total net revenue transferred along each edge
	"""
	save_paths = []
	for iter_,row in points_dataframe.iterrows():
		od_pair = ast.literal_eval(row['od_nodes'])
		pos0_i = graph.vs[node_dict[od_pair[0]]]
		pos1_i = graph.vs[node_dict[od_pair[1]]]
		# print (od_pair,pos0_i,pos1_i)
		if pos0_i != pos1_i:
			tons = row['min_croptons']
			vh_nums = math.ceil(1.0*tons/vehicle_wt)
			graph = add_igraph_generalised_costs_province_roads(graph,vh_nums,tons)
			path = graph.get_shortest_paths(pos0_i,pos1_i,weights='min_gcost',output="epath")
			get_path = [graph.es[n]['edge_id'] for n in path][0]

			# print ('path',get_path)
			# get the path edges, path length
			get_dist = sum([graph.es[n]['length'] for n in path][0])

			# get the minimum time and cost of transport along the path
			get_min_time = sum([graph.es[n]['min_time'] for n in path][0])
			get_min_gcost = sum([graph.es[n]['min_gcost'] for n in path][0])

			# recalculate for maximum values by changing the network costs
			# we get the same paths
			tons = row['max_croptons']
			vh_nums = math.ceil(1.0*tons/vehicle_wt)
			graph = add_igraph_generalised_costs_province_roads(graph,vh_nums,tons)

			# get the maximum time and cost of transport along the path
			get_max_time = sum([graph.es[n]['max_time'] for n in path][0])
			get_max_gcost = sum([graph.es[n]['max_gcost'] for n in path][0])

			print ('works',(od_pair,get_path,row['netrev_noagri']+row['min_agrirev'],row['netrev_noagri']+row['max_agrirev'],row['min_croptons'],row['max_croptons'],get_dist,get_min_time,get_max_time,get_min_gcost,get_max_gcost))

			save_paths.append((od_pair,get_path,row['netrev_noagri']+row['min_agrirev'],row['netrev_noagri']+row['max_agrirev'],
								row['min_croptons'],row['max_croptons'],get_dist,get_min_time,get_max_time,get_min_gcost,get_max_gcost))

	# 	try:
	# 		od_pair = ast.literal_eval(row['od_nodes'])
	# 		pos0_i = graph.vs[node_dict[od_pair[0]]]
	# 		pos1_i = graph.vs[node_dict[od_pair[1]]]
	# 		# print (od_pair,pos0_i,pos1_i)
	# 		if pos0_i != pos1_i:
	# 			tons = row['min_croptons']
	# 			vh_nums = math.ceil(1.0*tons/vechicle_wt)
	# 			G = add_igraph_generalised_costs_province_roads(G,vh_nums,tons)
	# 			path = graph.get_shortest_paths(pos0_i,pos1_i,weights='min_gcost',output="epath")

	# 			# get the path edges, path length
	# 			get_path = [graph.es[n]['edge_id'] for n in path][0]
	# 			get_dist = sum([graph.es[n]['length'] for n in path][0])

	# 			# get the minimum time and cost of transport along the path
	# 			get_min_time = sum([graph.es[n]['min_time'] for n in path][0])
	# 			get_min_gcost = sum([graph.es[n]['min_gcost'] for n in path][0])

	# 			# recalculate for maximum values by changing the network costs
	# 			# we get the same paths
	# 			tons = row['max_croptons']
	# 			vh_nums = math.ceil(1.0*tons/vechicle_wt)
	# 			G = add_igraph_generalised_costs_province_roads(G,vh_nums,tons)

	# 			# get the maximum time and cost of transport along the path
	# 			get_max_time = sum([graph.es[n]['max_time'] for n in path][0])
	# 			get_max_gcost = sum([graph.es[n]['max_gcost'] for n in path][0])

	# 			print ('works',(od_pair,get_path,row['netrev_noagri']+row['min_agrirev'],row['netrev_noagri']+row['max_agrirev'],row['min_croptons'],row['max_croptons'],get_dist,get_min_time,get_max_time,get_min_gcost,get_max_gcost))

	# 			save_paths.append((od_pair,get_path,row['netrev_noagri']+row['min_agrirev'],row['netrev_noagri']+row['max_agrirev'],
	# 								row['min_croptons'],row['max_croptons'],get_dist,get_min_time,get_max_time,get_min_gcost,get_max_gcost))

	# 	except:
	# 		print(iter_)

	# cols = ['od_nodes','edge_path','min_netrev','max_netrev','min_croptons','max_croptons','distance','min_time','max_time','min_gcost','max_gcost']
	# save_paths_df = pd.DataFrame(save_paths,columns = cols)
	# save_paths_df.to_excel(excel_writer,region_name,index = False)
	# excel_writer.save()
	# del save_paths_df

	# all_edges = [x['edge_id'] for x in graph.es]
	# all_edges_geom = [x['geometry'] for x in graph.es]

	# gdf_edges = gpd.GeoDataFrame(pd.DataFrame([all_edges,all_edges_geom]).T,crs='epsg:4326')
	# gdf_edges.columns = ['edge_id','geometry']

	# gdf_edges['min_netrev'] = 0
	# gdf_edges['max_netrev'] = 0
	# gdf_edges['min_croptons'] = 0
	# gdf_edges['max_croptons'] = 0

	# for path in save_paths:
	# 	gdf_edges.loc[gdf_edges['edge_id'].isin(path[1]),'min_netrev'] += path[2]
	# 	gdf_edges.loc[gdf_edges['edge_id'].isin(path[1]),'max_netrev'] += path[3]
	# 	gdf_edges.loc[gdf_edges['edge_id'].isin(path[1]),'min_netrev'] += path[4]
	# 	gdf_edges.loc[gdf_edges['edge_id'].isin(path[1]),'max_netrev'] += path[5]

	# if save_edges == True:
	# 	gdf_edges.to_file(os.path.join(output_path,'weighted_edges_district_center_flows_{}.shp'.format(region_name)))

def network_edges_assembly(points_dataframe,graph,tonnage_col,vehicle_wt,cost_criteria):
	"""
	Assign net revenue to roads assets in Vietnam

	Inputs are:
	start_points - GeoDataFrame of start points for shortest path analysis.
	end_points - GeoDataFrame of potential end points for shorest path analysis.
	G - iGraph network of the province.
	save_edges -

	Outputs are:
	Shapefile with all edges and the total net reveneu transferred along each edge
	GeoDataFrame of total net revenue transferred along each edge
	"""
	save_paths = []
	for iter_,row in points_dataframe.iterrows():
		try:
			od_pair = ast.literal_eval(row['od_nodes'])
			tons = row[tonnage_col]
			vh_nums = math.ceil(tons/vechicle_wt)
			G = add_igraph_generalised_costs_province_roads(G,vh_nums,tons)

			pos0_i = graph.vs[node_dict[od_pair[0]]]
			pos1_i = graph.vs[node_dict[od_pair[1]]]

			if pos0_i != pos1_i:
				path = graph.get_shortest_paths(pos0_i,pos1_i,weights=cost_criteria,output="epath")
				get_path = [graph.es[n]['edge_id'] for n in path][0]
				get_dist = sum([graph.es[n]['length'] for n in path][0])
				get_min_time = sum([graph.es[n]['min_time'] for n in path][0])
				get_max_time = sum([graph.es[n]['max_time'] for n in path][0])
				get_min_gcost = sum([graph.es[n]['min_gcost'] for n in path][0])
				get_max_gcost = sum([graph.es[n]['max_gcost'] for n in path][0])
				save_paths.append((od_pair,get_path,row['netrev_noagri']+row['max_agrirev'],row['netrev_noagri']+row['min_agrirev'],get_dist,get_min_time,get_max_time,get_min_gcost,get_max_gcost))
		except:
			print(iter_)

	return save_paths

def netrev_od_pairs(start_points,end_points):
	"""
	Assign net revenue to roads assets in Vietnam

	Inputs are:
	start_points - GeoDataFrame of start points for shortest path analysis.
	end_points - GeoDataFrame of potential end points for shorest path analysis.
	G - iGraph network of the province.
	save_edges -

	Outputs are:
	Shapefile with all edges and the total net reveneu transferred along each edge
	GeoDataFrame of total net revenue transferred along each edge
	"""
	save_paths = []
	for iter_,place in start_points.iterrows():
		try:
			closest_center = end_points.loc[end_points['OBJECTID']
			== place['NEAREST_C_CENTER']]['NEAREST_G_NODE'].values[0]

			get_od_pair = (place['NEAREST_G_NODE'],closest_center)
			save_paths.append((str(get_od_pair),1.0*place['netrev_agri']/12.0,1.0*place['netrev_noagri']/12.0))
		except:
			print(iter_)


	od_pairs_df = pd.DataFrame(save_paths,columns = ['od_nodes','netrev_agri','netrev_noagri'])
	od_pairs_df = od_pairs_df.groupby(['od_nodes'])['netrev_agri','netrev_noagri'].sum().reset_index()

	return od_pairs_df

def crop_od_pairs(start_points,end_points,crop_name):
	save_paths = []
	for iter_,place in start_points.iterrows():
		try:
			closest_center = end_points.loc[end_points['OBJECTID']
			== place['NEAREST_C_CENTER']]['NEAREST_G_NODE'].values[0]

			get_od_pair = (place['NEAREST_G_NODE'],closest_center)
			save_paths.append((str(get_od_pair),place['tons']))
		except:
			print(iter_)


	od_pairs_df = pd.DataFrame(save_paths,columns = ['od_nodes',crop_name])
	od_pairs_df = od_pairs_df.groupby(['od_nodes'])[crop_name].sum().reset_index()

	return od_pairs_df

def assign_minmax_rev_costs_crops(x,cost_dataframe,x_cols):
	'''
	crop_code	crop_name	min_cost_perton	max_cost_perton
	'''
	min_croprev = 0
	max_croprev = 0
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		if cost_param.crop_code in x_cols:
			min_croprev += 1.0*cost_param.min_cost_perton*x[cost_param.crop_code]
			max_croprev += 1.0*cost_param.max_cost_perton*x[cost_param.crop_code]

	return min_croprev, max_croprev

def assign_monthly_tons_crops(x,rice_prod_dist,x_cols):
	'''
	crop_code	crop_name	min_cost_perton	max_cost_perton
	'''
	min_croptons = 0
	max_croptons = 0
	for x_name in x_cols:
		if x_name == 'rice':
			min_croptons += (1.0*min(rice_prod_dist)*x[x_name])/12.0
			max_croptons += (1.0*max(rice_prod_dist)*x[x_name])/12.0
		else:
			min_croptons += (1.0*x[x_name])/365.0
			max_croptons += (1.0*x[x_name])/365.0

	return min_croptons, max_croptons

def assign_io_rev_costs_crops(x,cost_dataframe,rice_prod_dist,x_cols,ex_rate):
	'''
	crop_code	crop_name	min_cost_perton	max_cost_perton
	'''
	min_croprev = 0
	max_croprev = 0
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		if cost_param.crop_code in x_cols:
			if cost_param.crop_code == 'rice':
				min_croprev += (1.0*min(rice_prod_dist)*ex_rate*cost_param.est_net_rev*(x[cost_param.crop_code]/cost_param.tot_tons))/12.0
				max_croprev += (1.0*max(rice_prod_dist)*ex_rate*cost_param.est_net_rev*(x[cost_param.crop_code]/cost_param.tot_tons))/12.0
			else:
				min_croprev += 1.0/365.0*(ex_rate*cost_param.est_net_rev*(x[cost_param.crop_code]/cost_param.tot_tons))
				max_croprev += 1.0/365.0*(ex_rate*cost_param.est_net_rev*(x[cost_param.crop_code]/cost_param.tot_tons))

	return min_croprev, max_croprev

def line_endpoints(line):
	start = shapely.geometry.Point(line.coords[0])
	end = shapely.geometry.Point(line.coords[-1])
	return start, end


def line_end(line, point):
	start, end = line_endpoints(line)
	if start.equals(point):
		return 0
	elif end.equals(point):
		return 1

def cut(line, distance):
	# Cuts a line in two at a distance from its starting point
	if distance <= 0.0 or distance >= line.length:
		return [LineString(line)]
	coords = list(line.coords)
	for i, p in enumerate(coords):
		pd = line.project(Point(p))
		if pd == distance:
			return [
				LineString(coords[:i+1]),
				LineString(coords[i:])]
		if pd > distance:
			cp = line.interpolate(distance)
			return [
				LineString(coords[:i] + [(cp.x, cp.y)]),
				LineString([(cp.x, cp.y)] + coords[i:])]

def split_line_with_point(line, point):
	"""Split a line using a point

	- code directly similar to shapely.ops.split, with checks removed so that
		line must be split even if point doesn't intersect.
	"""
	distance_on_line = line.project(point)
	coords = list(line.coords)

	for j, p in enumerate(coords):
		pd = line.project(shapely.geometry.Point(p))
		if pd == distance_on_line:
			if j == 0 or j == len(coords) - 1:
				return [line]
			else:
				return [
					shapely.geometry.LineString(coords[:j+1]),
					shapely.geometry.LineString(coords[j:])
				]
		elif distance_on_line < pd:
			cp = line.interpolate(distance_on_line)
			ls1_coords = coords[:j]
			ls1_coords.append(cp.coords[0])
			ls2_coords = [cp.coords[0]]
			ls2_coords.extend(coords[j:])
			return [
				shapely.geometry.LineString(ls1_coords),
				shapely.geometry.LineString(ls2_coords)
			]

def split_line_with_points(line, points_list):
	"""Split a line using a point

	- code directly similar to shapely.ops.split, with checks removed so that
	line must be split even if point doesn't intersect.
	"""
	pt_tup_list = []
	for pts in points_list:
		pt_tup_list.append(tuple(list(pts) + [line.project(pts[2])]))

	pt_tup_list = [(p,w,x,y) for (p,w,x,y) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
	unique_fracs = list(set([y for (p,w,x,y) in pt_tup_list]))

	unique_fracs = sorted(unique_fracs)
	unique_pt_tup_list = []
	if len(unique_fracs) < len(pt_tup_list):
		for uf in unique_fracs:
			pt_tup_list_uf = [(p,w,x,y) for (p,w,x,y) in sorted(pt_tup_list, key=lambda pair: pair[1]) if y == uf]
			unique_pt_tup_list += [pt_tup_list_uf[0]]
	else:
		unique_pt_tup_list = pt_tup_list

	hits = [x for (p,w,x,y) in unique_pt_tup_list[1:-1]]
	hits_pts = [y for (p,w,x,y) in unique_pt_tup_list[1:-1]]
	# segments = [line]
	# for hit in hits:
	# 	new_segments = []
	# 	for segment in filter(lambda x: not x.is_empty, segments):
	# 		# add the newly split 2 lines or the same line if not split
	# 		new_segments.extend(split_line_with_point(segment, hit))
	# 		segments = new_segments

	new_segments = line
	segments = []
	last_sp = []
	for hit in range(len(hits)):
		l_sp = split_line_with_point(new_segments, hits[hit])
		if len(l_sp) > 1:
			new_segments = l_sp[1]
		else:
			new_segments = l_sp[0]

		segments.append(l_sp[0])

		if hit == (len(hits)-1):
			last_sp = l_sp

	# print ('last',last_sp)

	if len(last_sp) > 1:
		segments.append(last_sp[1])
	else:
		print (unique_pt_tup_list)

	segments = list(segments)
	line_dict = []
	if len(segments) < len(unique_pt_tup_list):
		for s in range(len(segments)):
			line_dict.append({'from_node':unique_pt_tup_list[s][0],'to_node':unique_pt_tup_list[s+1][0],'geometry':segments[s],'from_geom':unique_pt_tup_list[s][2],'to_geom':unique_pt_tup_list[s+1][2]})

	return line_dict

def split_lines_at_many_nodes(nodes, node_index, ways):
	"""Split all ways at station nodes
	"""
	max_id = 0
	split_ways = {}
	split_ways_index = index.Index()
	for i, way in ways.items():
		check_ids = list(node_index.intersection(way['geometry'].bounds))
		hits = []

		for node_i in check_ids:
			node = nodes[node_i]
			intersection = way['geometry'].intersection(node['geometry'])

			if not intersection:
				intersection = way['geometry'].intersection(node['geometry'].buffer(0.0000000001))

			if not intersection:
				continue
			elif intersection.geometryType() == 'Point':
				hits.append(intersection)
			elif intersection.geometryType() == 'MultiPoint':
				hits.extend(point for point in intersection)
			elif intersection.geometryType() == 'LineString':
				start_point = shapely.geometry.Point(intersection.coords[0])
				hits.append(start_point)
			elif intersection.geometryType() == 'GeometryCollection':
				hits.extend(geom for geom in list(intersection) if geom.geometryType() == 'Point')
			else:
				print("Unhandled intersection type:")
				print(way)
				print(node)
				print(intersection)
				exit()

		segments = [way['geometry']]
		if hits:
			# segments = shapely.ops.split(way['geom'], shapely.geometry.MultiPoint(hits))
			for hit in hits:
				new_segments = []
				for segment in filter(lambda x: not x.is_empty, segments):
					# add the newly split 2 lines or the same line if not split
					new_segments.extend(split_line_with_point(segment, hit))
					segments = new_segments

		for segment in list(segments):
			split_ways[max_id] = {
				'edge_id': edge_name + str(max_id),
				'from_node': from_node,
				'to_node':to_node,
				'geometry': shapely.geometry.mapping(segment),
			}
			split_ways_index.insert(max_id, segment.bounds)
			max_id += 1
	return split_ways, split_ways_index

def modify_network_with_new_points(points_dataframe,network_edge_dataframe,network_node_dataframe,edge_id_column,node_id_column):
	max_node_count = max([int(x.split('_')[1]) for x in network_node_dataframe[node_id_column].values.tolist()])
	node_string = [x.split('_')[0] for x in network_node_dataframe[node_id_column].values.tolist()][0]

	max_edge_count = max([int(x.split('_')[1]) for x in network_edge_dataframe[edge_id_column].values.tolist()])
	edge_string = [x.split('_')[0] for x in network_edge_dataframe[edge_id_column].values.tolist()][0]


	points_dataframe = points_dataframe.set_index(edge_id_column)
	network_edge_dataframe = network_edge_dataframe.set_index(edge_id_column)
	network_node_dataframe = network_node_dataframe.set_index(node_id_column)

	all_edge_list = []
	all_node_list = []
	edge_remove_list = []
	unique_edge_values = list(set(points_dataframe.index.values.tolist()))
	for unique_edge in unique_edge_values:
		points = points_dataframe.loc[[unique_edge],'geometry'].values.tolist()
		points_nums = list(np.array([max_node_count]*len(points)) + np.arange(1,len(points)+1,1))
		max_node_count = points_nums[-1]
		points_ids = [node_string + '_' + str(x) for x in points_nums]

		edge_vals = dict(network_edge_dataframe.loc[unique_edge])
		points += [network_node_dataframe.loc[edge_vals['from_node'],'geometry'],network_node_dataframe.loc[edge_vals['to_node'],'geometry']]
		points_ids += [edge_vals['from_node'],edge_vals['to_node']]
		points_nums += [int(edge_vals['from_node'].split('_')[1]),int(edge_vals['to_node'].split('_')[1])]

		edge_list = split_line_with_points(edge_vals['geometry'], list(zip(points_ids,points_nums,points)))
		if edge_list:
			edge_list_df = pd.DataFrame(edge_list,columns = ['from_node','to_node','geometry','from_geom','to_geom'])

			nodes_inc = list(zip(edge_list_df['from_node'].values.tolist(),edge_list_df['from_geom'].values.tolist())) + list(zip(edge_list_df['to_node'].values.tolist(),edge_list_df['to_geom'].values.tolist()))
			node_list_df = pd.DataFrame(nodes_inc,columns = ['node_id','geometry'])
			node_list_df = node_list_df.drop_duplicates(subset=['node_id'], keep='first')
			all_node_list.append(node_list_df)

			edge_list_df.drop('from_geom',axis=1,inplace=True)
			edge_list_df.drop('to_geom',axis=1,inplace=True)
			# nodes_unique = list(set([x[0] for x in nodes_inc]))
			# for nu in nodes_unique:
			# 	geom_unique = [x[1] for x in nodes_inc]
			line_nums = list(np.array([max_edge_count]*len(edge_list)) + np.arange(1,len(edge_list)+1,1))
			max_edge_count = line_nums[-1]
			edge_list_df['g_id'] =  line_nums
			edge_list_df['edge_id'] = [edge_string + '_' + str(x) for x in line_nums]
			for edge_key,edge_attr in edge_vals.items():
				if edge_key not in ['edge_id','g_id','from_node','to_node','geometry']:
					edge_list_df[edge_key] = edge_attr

			all_edge_list.append(edge_list_df)
			edge_remove_list.append(unique_edge)
		else:
			print(unique_edge)
		# points_gpd = gpd.DataFrame(pd.DataFrame(list(zip(points_ids,points_nums,points)),columns = ['node_id','gid','geometry']),crs = 'epsg:4326')
		# edge_start_point, edge_end_point = line_endpoints(edge_vals['geometry'])
		# print (edge_list_df)

	edge_list_df = gpd.GeoDataFrame(pd.concat(all_edge_list, axis=0, sort = 'False', ignore_index=True),crs = 'epsg:4326')
	# out_shp = os.path.join(data_path,'Multi','multi_edges','roads_multi_test_edges.shp')
	# edge_list_df.to_file(out_shp)

	node_list_df = gpd.GeoDataFrame(pd.concat(all_node_list, axis=0, sort = 'False', ignore_index=True),crs = 'epsg:4326')
	print (node_list_df)
	# out_shp = os.path.join(data_path,'Multi','multi_edges','roads_multi_test_nodes.shp')
	# node_list_df.to_file(out_shp)

	network_edge_dataframe = network_edge_dataframe.reset_index()
	update_edges = gpd.GeoDataFrame(pd.concat([network_edge_dataframe,edge_list_df], axis=0, sort = 'False', ignore_index=True),crs = 'epsg:4326')
	update_edges = update_edges[~update_edges['edge_id'].isin(edge_remove_list)]
	out_shp = os.path.join(data_path,'Roads','national_roads','national_network_edges.shp')
	update_edges.to_file(out_shp)

	network_node_dataframe = network_node_dataframe.reset_index()
	update_nodes = gpd.GeoDataFrame(pd.concat([network_node_dataframe,node_list_df], axis=0, sort = 'False', ignore_index=True),crs = 'epsg:4326')
	update_nodes = update_nodes.drop_duplicates(subset=['node_id'], keep='first')
	out_shp = os.path.join(data_path,'Roads','national_roads','national_network_nodes.shp')
	update_nodes.to_file(out_shp)

if __name__ == '__main__':

	data_path,calc_path,output_path = load_config()['paths']['data'],load_config()['paths']['calc'],load_config()['paths']['output']

	# edges_in = os.path.join(data_path,'Roads','national_network_edges_v1','national_network_edges_v1.shp')
	# edges_in = os.path.join(data_path,'Roads','traffic_count','road_network.shp')
	# edges_in = os.path.join(data_path,'Roads','traffic_count','national_network_edges_v1_edges.shp')
	# edges = gpd.read_file(edges_in)
	# edges.columns = map(str.lower, edges.columns)
	# # get the right linelength
	# edges['length'] = edges.geometry.apply(line_length)
	# length_attr = list(zip(edges['g_id'].values.tolist(),edges['length'].values.tolist()))

	# edges_in = os.path.join(data_path,'Roads','traffic_count','road_network.shp')
	# edges = gpd.read_file(edges_in)
	# vals = list(zip(edges['G_ID'].values.tolist(),edges['vehicle_co'].values.tolist()))

	# edges_in = os.path.join(data_path,'Roads','traffic_count','national_network_edges_v1_edges.shp')
	# edges = gpd.read_file(edges_in)
	# # edges.columns = map(str.lower, edges.columns)
	# edges['vehicle_co'] = 0

	# for v in vals:
	# 	edges.loc[edges['L_ID'] == v[0], 'vehicle_co'] = v[1]

	# edges.to_file(os.path.join(data_path,'Roads','national_roads','national_network_edges.shp'))

	# nodes_in = os.path.join(data_path,'Roads','traffic_count','national_network_edges_v1_nodes.shp')
	# nodes = gpd.read_file(nodes_in)

	# nodes.to_file(os.path.join(data_path,'Roads','national_roads','national_network_nodes.shp'))

	# no_geom = []
	# for iter_,vals in edges.iterrows():
	# 	# if vals['geometry'].geom_type != 'LineString':
	# 	# 	print (vals['edge_id'])

	# 	if vals['geometry'] is None:
	# 		print (vals['edge_id'])
	# 		no_geom.append(vals['edge_id'])

	# print (no_geom)
	# print (edges)
	# edges = edges.reindex(list(edges.columns)[2:]+list(edges.columns)[:2],axis=1)
	# G = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:])

	# graph = G.clusters().giant()

	'''
	vehicle_id, edge_path, time_stamp
	'''
	'''
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	Traffic speed assignment script
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	'''
	# edges_in = os.path.join(data_path,'Roads','traffic_count','road_network.shp')
	# edges = gpd.read_file(edges_in)
	# edges.columns = map(str.lower, edges.columns)
	# # get the right linelength
	# edges['length'] = edges.geometry.apply(line_length)
	# length_attr = list(zip(edges['g_id'].values.tolist(),edges['length'].values.tolist()))

	# routes_in = os.path.join(data_path,'Roads','routes_collected','routes.csv')
	# routes_df = pd.read_csv(routes_in)
	# # print (len(routes_df.index))
	# # edge_paths = list(set(routes_df['edge_path'].values.tolist()))
	# # # print (len(edge_paths))
	# # edge_paths = [ast.literal_eval(e) for e in edge_paths if len(ast.literal_eval(e)) > 0]
	# # print (len(edge_paths))
	# # print (edge_paths[0:10])

	# edge_speeds = []
	# counts = 0
	# for iter_,vals in routes_df.iterrows():
	# 	edge_path = ast.literal_eval(vals['edge_path'])
	# 	time_stamp = ast.literal_eval(vals['time_stamp'])
	# 	if len(edge_path) > 1:
	# 		# tot_dist = sum([l[1] for l in length_attr if l[0] in edge_path])
	# 		# tot_time = 1.0*(time_stamp[-1] - time_stamp[0])
	# 		# avg_speed = 3600.0*tot_dist/tot_time
	# 		# print ('averages',tot_dist,tot_time,avg_speed)
	# 		for e in range(len(edge_path)-1):
	# 			time_diff = 1.0*(time_stamp[e+1] - time_stamp[e])
	# 			# if time_diff > 0 and time_diff < 1000.0:
	# 			if time_diff > 0:
	# 				distance = sum([l[1] for l in length_attr if l[0] in (edge_path[e],edge_path[e+1])])
	# 				# distance = sum([l[1] for l in length_attr if l[0] == edge_path[e]])
	# 				edge_l = [l[1] for l in length_attr if l[0] == edge_path[e]] + [l[1] for l in length_attr if l[0] == edge_path[e+1]]
	# 				speed = 3600.0*distance/time_diff
	# 				# print (edge_path[e],edge_path[e+1],edge_l,time_diff,speed)
	# 				# edge_speeds.append((edge_path[e],speed))
	# 				# edge_speeds.append((edge_path[e+1],speed))
	# 				if speed >= 20 and speed <= 120:
	# 					edge_speeds.append((edge_path[e],speed))
	# 					edge_speeds.append((edge_path[e+1],speed))
	# 					# print ('accepted',edge_path[e],edge_path[e+1],edge_l,time_diff,speed)




	# 	# counts +=1
	# 	# if counts > 100:
	# 	# 	break

	# 	print ('Done with iteration',iter_)

	# del routes_df

	# edge_speeds_df = pd.DataFrame(edge_speeds,columns = ['g_id','speed'])

	# # edge_speeds_df_min = edge_speeds_df.groupby(['g_id'])['speed'].min().reset_index()
	# # edge_speeds_df_min.rename(columns={'speed': 'min_speed'}, inplace=True)
	# # edges = pd.merge(edges,edge_speeds_df_min,how='left', on=['g_id']).fillna(0)
	# # del edge_speeds_df_min

	# # edge_speeds_df_max = edge_speeds_df.groupby(['g_id'])['speed'].max().reset_index()
	# # edge_speeds_df_max.rename(columns={'speed': 'max_speed'}, inplace=True)
	# # edges = pd.merge(edges,edge_speeds_df_max,how='left', on=['g_id']).fillna(0)
	# # del edge_speeds_df_max

	# # edge_speeds_df_median = edge_speeds_df.groupby(['g_id'])['speed'].median().reset_index()
	# # edge_speeds_df_median.rename(columns={'speed': 'md_speed'}, inplace=True)
	# # edges = pd.merge(edges,edge_speeds_df_median,how='left', on=['g_id']).fillna(0)
	# # del edge_speeds_df_median

	# edge_speeds_df_mean = edge_speeds_df.groupby(['g_id'])['speed'].mean().reset_index()
	# edge_speeds_df_mean.rename(columns={'speed': 'mean_speed'}, inplace=True)
	# edges = pd.merge(edges,edge_speeds_df_mean,how='left', on=['g_id']).fillna(0)
	# del edge_speeds_df_mean


	# edge_speeds_df_std = edge_speeds_df.groupby(['g_id'])['speed'].std().reset_index()
	# edge_speeds_df_std.rename(columns={'speed': 'std_speed'}, inplace=True)
	# edges = pd.merge(edges,edge_speeds_df_std,how='left', on=['g_id']).fillna(0)
	# del edge_speeds_df_std
	# del edge_speeds_df

	# # edge_speeds_df = edges[['g_id','min_speed','max_speed','mean_speed','md_speed','std_speed']]
	# edge_speeds_df = edges[['g_id','mean_speed','std_speed']]
	# del edges
	# edge_speeds = list(edge_speeds_df.itertuples(index=False))
	# # edges.to_file(os.path.join(data_path,'Roads','national_roads','national_network_edges_v2.shp'))

	# edges_in = os.path.join(data_path,'Roads','national_roads','national_network_edges.shp')
	# edges = gpd.read_file(edges_in)
	# # edges.columns = map(str.lower, edges.columns)
	# edges['zscore'] = list(stats.zscore(edges['vehicle_co']))
	# # edges['min_speed'] = 0
	# # edges['max_speed'] = 0
	# edges['mean_speed'] = 0
	# # edges['md_speed'] = 0
	# edges['std_speed'] = 0
	# edges['est_speed'] = 0


	# for e in edge_speeds:
	# 	# edges.loc[edges['L_ID'] == e.g_id, 'min_speed'] = e.min_speed
	# 	# edges.loc[edges['L_ID'] == e.g_id, 'max_speed'] = e.max_speed
	# 	edges.loc[edges['L_ID'] == e.g_id, 'mean_speed'] = e.mean_speed
	# 	# edges.loc[edges['L_ID'] == e.g_id, 'md_speed'] = e.md_speed
	# 	edges.loc[edges['L_ID'] == e.g_id, 'std_speed'] = e.std_speed

	# 	edges.loc[edges['L_ID'] == e.g_id, 'est_speed'] = e.mean_speed + edges.loc[edges['L_ID'] == e.g_id, 'zscore']*e.std_speed

	# edges.loc[edges['est_speed'] > 120.0,'est_speed'] = 120.0

	# edges.to_file(os.path.join(data_path,'Roads','national_roads','national_network_edges.shp'))

	'''
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	Traffic speed assignment script
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	'''

	# all_edges = [x['edge_id'] for x in graph.es]
	# all_edges_geom = [x['geometry'] for x in graph.es]

	# gdf_edges = gpd.GeoDataFrame(pd.DataFrame([all_edges,all_edges_geom]).T,crs='epsg:4326')
	# gdf_edges.columns = ['edge_id','geometry']

	# gdf_edges.to_file(os.path.join(output_path,'national_roads','connect_graph_v1.shp'))


	# truck_unit_wt = 20.0
	# # provinces to consider
	# province_list = ['Lao Cai','Binh Dinh','Thanh Hoa']
	# province_terrian = ['mountain','flat','flat']
	# # province_list = ['Thanh Hoa']
	# district_committe_names = ['district_people_committee_points_lao_cai.shp',
	# 						'district_province_peoples_committee_point_binh_dinh.shp',
	# 						'district_people_committee_points_thanh_hoa.shp']

	# exchange_rate = 1.05*(1000000/21000)

	# shp_output_path = os.path.join(output_path,'flow_mapping_shapefiles')
	# flow_output_excel = os.path.join(output_path,'flow_mapping_paths','province_roads_district_center_flow_paths.xlsx')
	# excl_wrtr = pd.ExcelWriter(flow_output_excel)

	# rd_prop_file = os.path.join(data_path,'Roads','road_properties','road_properties.xlsx')
	# province_path = os.path.join(data_path,'Vietnam_boundaries','who_boundaries','who_provinces.shp')
	# population_points_in = os.path.join(data_path,'Points_of_interest','population_points.shp')
	# commune_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','commune_level_stats.shp')

	# crop_data_path = os.path.join(data_path,'Agriculture_crops','crop_data')
	# rice_month_file = os.path.join(data_path,'rice_atlas_vietnam','rice_production.shp')
	# crop_month_fields = ['P_Jan','P_Feb','P_Mar','P_Apr','P_May','P_Jun','P_Jul','P_Aug','P_Sep','P_Oct','P_Nov','P_Dec']
	# crop_names = ['rice','cash','cass','teas','maiz','rubb','swpo','acof','rcof','pepp']


	# # for prn in range(len(province_list)):
	# for prn in range(0,1):
	# 	province_ods_df = []
	# 	province = province_list[prn]
	# 	# set all paths for all input files we are going to use
	# 	province_name = province.replace(' ','').lower()

	# 	edges_in = os.path.join(data_path,'Roads','{}_roads'.format(province_name),'vietbando_{}_edges.shp'.format(province_name))
	# 	nodes_in = os.path.join(data_path,'Roads','{}_roads'.format(province_name),'vietbando_{}_nodes.shp'.format(province_name))

	# 	# commune_center_in = os.path.join(data_path,'Points_of_interest',district_committe_names[prn])

	# 	# # path_width_table = os.path.join(data_path,'Roads','road_properties','road_properties.xlsx')

	# 	# # load provinces and get geometry of the right province
	# 	# provinces = gpd.read_file(province_path)
	# 	# provinces = provinces.to_crs({'init': 'epsg:4326'})
	# 	# province_geom = provinces.loc[provinces.NAME_ENG == province].geometry.values[0]

	# 	# # clip all the populations to the province
	# 	# prov_pop = gdf_clip(population_points_in,province_geom)
	# 	# prov_commune_center = gdf_clip(commune_center_in,province_geom)
	# 	# if 'OBJECTID' not in prov_commune_center.columns.values.tolist():
	# 	# 	prov_commune_center['OBJECTID'] = prov_commune_center.index

	# 	# prov_communes = gdf_clip(commune_path,province_geom)

	# 	# # load nodes of the network
	# 	# nodes = gpd.read_file(nodes_in)
	# 	# nodes = nodes.to_crs({'init': 'epsg:4326'})
	# 	# sindex_nodes = nodes.sindex

	# 	# # get revenue values for each village
	# 	# # first create sindex of all villages to count number of villages in commune
	# 	# prov_pop_sindex = prov_pop.sindex

	# 	# # create new column in prov_communes with amount of villages
	# 	# prov_communes['n_villages'] = prov_communes.geometry.apply(lambda x: count_points_in_polygon(x,prov_pop_sindex))
	# 	# prov_communes['netrev_village'] = exchange_rate*(prov_communes['netrevenue']*prov_communes['nfirm'])/prov_communes['n_villages']
	# 	# # also get the net revenue of the agriculture sector which is called nongnghiep
	# 	# prov_communes['netrev_village_agri'] = 1.0/365.0*(prov_communes['nongnghiep']*prov_communes['netrev_village'])
	# 	# prov_communes['netrev_village_noagri'] = 1.0/365.0*(prov_communes['netrev_village'] - prov_communes['netrev_village_agri'])


	# 	# commune_sindex = prov_communes.sindex
	# 	# # give each village a net revenue based on average per village in commune
	# 	# prov_pop['netrev_agri'] = prov_pop.geometry.apply(lambda x: extract_value_from_gdf(x,commune_sindex,prov_communes,'netrev_village_agri'))
	# 	# prov_pop['netrev_noagri'] = prov_pop.geometry.apply(lambda x: extract_value_from_gdf(x,commune_sindex,prov_communes,'netrev_village_noagri'))


	# 	# # get nearest node in network for all start and end points
	# 	# prov_pop['NEAREST_G_NODE'] = prov_pop.geometry.apply(lambda x: get_nearest_node(x,sindex_nodes,nodes,'NODE_ID'))
	# 	# prov_commune_center['NEAREST_G_NODE'] = prov_commune_center.geometry.apply(lambda x: get_nearest_node(x,sindex_nodes,nodes,'NODE_ID'))

	# 	# # prepare for shortest path routing, we'll use the spatial index of the centers
	# 	# # to find the nearest center for each population point
	# 	# sindex_commune_center = prov_commune_center.sindex
	# 	# prov_pop['NEAREST_C_CENTER'] = prov_pop.geometry.apply(lambda x: get_nearest_node(x,sindex_commune_center,prov_commune_center,'OBJECTID'))

	# 	# # find all OD pairs of the revenues
	# 	# netrev_ods = netrev_od_pairs(prov_pop,prov_commune_center)
	# 	# province_ods_df.append(netrev_ods)

	# 	# # find the crop production months for the province
	# 	# rice_prod_months = gpd.read_file(rice_month_file)
	# 	# rice_prod_months = rice_prod_months.loc[rice_prod_months.SUB_REGION == province]
	# 	# rice_prod_months = rice_prod_months[crop_month_fields].values.tolist()
	# 	# rice_prod_months = np.array(rice_prod_months[0])/sum(rice_prod_months[0])
	# 	# rice_prod_months = rice_prod_months[rice_prod_months > 0]
	# 	# rice_prod_months = rice_prod_months.tolist()



	# 	# # all the crop OD pairs
	# 	# for file in os.listdir(crop_data_path):
	# 	# 	if file.endswith(".tif") and 'spam_p' in file.lower().strip():
	# 	# 		fpath = os.path.join(crop_data_path, file)
	# 	# 		crop_name = [cr for cr in crop_names if cr in file.lower().strip()][0]
	# 	# 		outCSVName = os.path.join(output_path,'crop_flows','crop_concentrations.csv')
	# 	# 		subprocess.run(["gdal2xyz.py",'-csv', fpath,outCSVName])

	# 	# 		'''Load points and convert to geodataframe with coordinates'''
	# 	# 		load_points = pd.read_csv(outCSVName,header=None,names=['x','y','tons'],index_col=None)
	# 	# 		load_points = load_points[load_points['tons'] > 0]

	# 	# 		geometry = [Point(xy) for xy in zip(load_points.x, load_points.y)]
	# 	# 		load_points = load_points.drop(['x', 'y'], axis=1)
	# 	# 		crs = {'init': 'epsg:4326'}
	# 	# 		crop_points = gpd.GeoDataFrame(load_points, crs=crs, geometry=geometry)

	# 	# 		del load_points

	# 	# 		# clip all to province
	# 	# 		prov_crop = gdf_geom_clip(crop_points,province_geom)

	# 	# 		if len(prov_crop.index) > 0:
	# 	# 			prov_crop_sindex = prov_crop.sindex
	# 	# 			prov_crop['NEAREST_G_NODE'] = prov_crop.geometry.apply(lambda x: get_nearest_node(x,sindex_nodes,nodes,'NODE_ID'))
	# 	# 			sindex_commune_center = prov_commune_center.sindex
	# 	# 			prov_crop['NEAREST_C_CENTER'] = prov_crop.geometry.apply(lambda x: get_nearest_node(x,sindex_commune_center,prov_commune_center,'OBJECTID'))

	# 	# 			crop_ods = crop_od_pairs(prov_crop,prov_commune_center,crop_name)
	# 	# 			province_ods_df.append(crop_ods)

	# 	# 			print ('Done with crop {0} in province {1}'.format(crop_name, province_name))

	# 	# all_ods = pd.concat(province_ods_df, axis=0, sort = 'False', ignore_index=True).fillna(0)

	# 	# all_ods_crop_cols = [c for c in all_ods.columns.values.tolist() if c in crop_names]
	# 	# all_ods['crop_tot'] = all_ods[all_ods_crop_cols].sum(axis = 1)

	# 	# all_ods_val_cols = [c for c in all_ods.columns.values.tolist() if c != 'od_nodes']
	# 	# all_ods = all_ods.groupby(['od_nodes'])[all_ods_val_cols].sum().reset_index()

	# 	# all_ods['croptons'] = all_ods.apply(lambda x: assign_monthly_tons_crops(x,rice_prod_months,all_ods_crop_cols),axis = 1)
	# 	# all_ods[['min_croptons', 'max_croptons']] = all_ods['croptons'].apply(pd.Series)
	# 	# all_ods.drop('croptons',axis=1,inplace=True)

	# 	# cost_values_df = pd.read_excel(os.path.join(crop_data_path,'crop_unit_costs.xlsx'),sheet_name ='io_rev')
	# 	# all_ods['croprev'] = all_ods.apply(lambda x: assign_io_rev_costs_crops(x,cost_values_df,rice_prod_months,all_ods.columns.values.tolist(),exchange_rate),axis = 1)
	# 	# all_ods[['min_agrirev', 'max_croprev']] = all_ods['croprev'].apply(pd.Series)
	# 	# all_ods.drop('croprev',axis=1,inplace=True)
	# 	# all_ods['max_agrirev'] = all_ods[['max_croprev','netrev_agri']].max(axis = 1)
	# 	# all_ods.drop(['max_croprev','netrev_agri'],axis=1,inplace=True)

	# 	# all_ods.to_csv(os.path.join(output_path,'{}_ods_1.csv'.format(province_name)),index = False)

	# 	# all_od_pairs = all_ods['od_nodes'].values.tolist()
	# 	# all_od_pairs = [ast.literal_eval(ods) for ods in all_od_pairs]

	# 	# common_pts = list(set([a[1] for a in all_od_pairs]))
	# 	# for source in common_pts:
	# 	# 	targets = [a[0] for a in all_od_pairs if a[1] == source]

	# 	all_ods = pd.read_csv(os.path.join(output_path,'{}_ods_1.csv'.format(province_name)))
	# 	# print (all_ods)
	# 	G = province_shapefile_to_network(edges_in,province_terrian[prn],rd_prop_file)
	# 	nodes_name = np.asarray([x['name'] for x in G.vs])
	# 	nodes_index = np.asarray([x.index for x in G.vs])
	# 	node_dict = dict(zip(nodes_name,nodes_index))
	# 	# print (node_dict)
	# 	# # network_od_paths_assembly(all_ods,G,truck_unit_wt)
	# 	# # print (all_ods)
	# 	network_od_paths_assembly(all_ods,node_dict,G,truck_unit_wt,province_name,save_edges = True,output_path =shp_output_path,excel_writer =excl_wrtr)
	'''
	++++++++++++++++++++++++++++
	Voronoi test
	++++++++++++++++++++++++++++
	'''
	# province_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','province_level_stats.shp')
	# commune_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','commune_level_stats.shp')

	# # load provinces and get geometry of the right province
	# provinces = gpd.read_file(province_path)
	# provinces = provinces.to_crs({'init': 'epsg:4326'})
	# sindex_provinces = provinces.sindex

	# # load provinces and get geometry of the right province
	# communes = gpd.read_file(commune_path)
	# communes = communes.to_crs({'init': 'epsg:4326'})
	# # sindex_communes = communes.sindex

	# modes_file_paths = [('Roads','national_roads')]
	# # modes_file_paths = [('Railways','national_rail')]
	# for m in range(len(modes_file_paths)):
	# 	mode_data_path = os.path.join(data_path,modes_file_paths[m][0],modes_file_paths[m][1])
	# 	for file in os.listdir(mode_data_path):
	# 		try:
	# 			if file.endswith(".shp") and 'edges' in file.lower().strip():
	# 				edges_in = os.path.join(mode_data_path, file)
	# 			if file.endswith(".shp") and 'nodes' in file.lower().strip():
	# 				nodes_in = os.path.join(mode_data_path, file)
	# 				print (os.path.join(mode_data_path, file))
	# 		except:
	# 			print ('Network nodes and edge files necessary')


	# 	# load nodes of the network
	# 	nodes = gpd.read_file(nodes_in)
	# 	nodes = nodes.to_crs({'init': 'epsg:4326'})
	# 	sindex_nodes = nodes.sindex

	# 	node_list = nodes['NODE_ID'].values.tolist()
	# 	print (len(node_list))

	# 	xy_list = []
	# 	for iter_,values in nodes.iterrows():
	# 		# print (list(values.geometry.coords))
	# 		xy = list(values.geometry.coords)
	# 		xy_list += [list(xy[0])]

	# 	print (len(xy_list))
	# 	vor = Voronoi(np.array(xy_list))
	# 	regions, vertices = voronoi_finite_polygons_2d(vor)
	# 	min_x = vor.min_bound[0] - 0.1
	# 	max_x = vor.max_bound[0] + 0.1
	# 	min_y = vor.min_bound[1] - 0.1
	# 	max_y = vor.max_bound[1] + 0.1

	# 	mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
	# 	bounded_vertices = np.max((vertices, mins), axis=0)
	# 	maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
	# 	bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

	# 	box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
	# 	# colorize
	# 	poly_list = []
	# 	for region in regions:
	# 		polygon = vertices[region]
	# 		# Clipping polygon
	# 		poly = Polygon(polygon)
	# 		poly = poly.intersection(box)
	# 		polygon = [p for p in poly.exterior.coords]
	# 		poly_list.append(poly)
	# 	# voronoi_plot_2d(vor)
	# 	# plt.show()
	# 	# lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
	# 	# # lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices]
	# 	# poly_list = []
	# 	# for poly in shapely.ops.polygonize(lines):
	# 	# 	# print (poly.wkt)
	# 	# 	poly_list.append(poly)
	# 	# 	if poly.geom_type != 'Polygon':
	# 	# 		print ('not polygon',poly.geom_type)

	# 	print (len(poly_list))
	# 	print (poly_list)

	# 	poly_index = list(np.arange(0,len(poly_list),1))
	# 	poly_df = pd.DataFrame(list(zip(poly_index,poly_list)),columns = ['gid','geometry'])
	# 	gdf_voronoi = gpd.GeoDataFrame(poly_df,crs='epsg:4326')
	# 	gdf_voronoi['node_id'] = gdf_voronoi.apply(lambda x: extract_nodes_within_gdf(x,nodes,'NODE_ID'),axis = 1)
	# 	gdf_voronoi['population'] = 0
	# 	gdf_voronoi = assign_value_in_area_proportions(communes,gdf_voronoi,'population')

	# 	# gdf_voronoi.columns = ['node_id','geometry']
	# 	print (gdf_voronoi)
	# 	gdf_voronoi.to_csv(os.path.join(output_path,'Voronoi','test.csv'))

	# 	gdf_voronoi.to_file(os.path.join(output_path,'Voronoi','test.shp'))

	'''
	++++++++++++++++++++++++++++
	Voronoi test
	++++++++++++++++++++++++++++
	'''

	'''
	+++++++++++++++++++++++++++++++++++++++++++++++++
	Merge values from different excel sheets in one
	+++++++++++++++++++++++++++++++++++++++++++++++++
	'''

	# modes = ['road','rail','air','inland','coastal']

	# od_excel = os.path.join(output_path,'flow_mapping_paths','national_scale_od_matrix.xlsx')
	# output_excel = os.path.join(output_path,'flow_mapping_paths','national_scale_od_matrix_total.xlsx')
	# excl_wrtr = pd.ExcelWriter(output_excel)
	# region_total = []
	# for m in modes:
	# 	region_total.append(pd.read_excel(od_excel,sheet_name = m))

	# all_ods = pd.concat(region_total, axis=0, sort = 'False', ignore_index=True).fillna(0)
	# all_ods_val_cols = [c for c in all_ods.columns.values.tolist() if c not in ('o_region','d_region')]
	# print (all_ods_val_cols)
	# all_ods_regions = all_ods.groupby(['o_region','d_region'])[all_ods_val_cols].sum().reset_index()
	# all_ods_regions.to_excel(excl_wrtr,'total',index = False)
	# excl_wrtr.save()

	'''
	+++++++++++++++++++++++++++++++++++++++++++++++++
	Merge values from different excel sheets in one
	+++++++++++++++++++++++++++++++++++++++++++++++++
	'''

	'''
	+++++++++++++++++++++++++++++++++++++++++++++++++
	create multi-modal edges
	+++++++++++++++++++++++++++++++++++++++++++++++++
	'''

	# from_sector = [('Waterways','waterways'),('Railways','national_rail')]
	# to_sector = [('Roads','national_roads'),('Railways','national_rail')]

	# all_multi = []
	# all_multi_pts = []
	# for f_s in from_sector:
	# 	for t_s in to_sector:
	# 		if f_s[0] != t_s[0]:
	# 			mode_data_path = os.path.join(data_path,f_s[0],f_s[1])
	# 			for file in os.listdir(mode_data_path):
	# 				try:
	# 					if file.endswith(".shp") and 'nodes' in file.lower().strip():
	# 						from_nodes_in = os.path.join(mode_data_path, file)
	# 				except:
	# 					print ('Network nodes and edge files necessary')

	# 			mode_data_path = os.path.join(data_path,t_s[0],t_s[1])

	# 			for file in os.listdir(mode_data_path):
	# 				try:
	# 					if file.endswith(".shp") and 'edges' in file.lower().strip():
	# 						to_edges_in = os.path.join(mode_data_path, file)

	# 					if file.endswith(".shp") and 'nodes' in file.lower().strip():
	# 						to_nodes_in = os.path.join(mode_data_path, file)
	# 				except:
	# 					print ('Network nodes and edge files necessary')


	# 			from_nodes_df = gpd.read_file(from_nodes_in).fillna(0)
	# 			from_nodes_df.columns = map(str.lower, from_nodes_df.columns)
	# 			if f_s[0] == 'Waterways':
	# 				from_nodes_df = from_nodes_df[(from_nodes_df['port_type'] == 'inland') | (from_nodes_df['port_type'] == 'sea')]
	# 			elif f_s[0] == 'Railways':
	# 				from_nodes_df = from_nodes_df[from_nodes_df['objectid'] >  0]

	# 			print (from_nodes_df)
	# 			to_nodes_df = gpd.read_file(to_nodes_in)
	# 			to_nodes_df.columns = map(str.lower, to_nodes_df.columns)
	# 			if t_s[0] == 'Waterways':
	# 				to_nodes_df = to_nodes_df[(to_nodes_df['port_type'] == 'inland') | (to_nodes_df['port_type'] == 'sea')]
	# 			elif t_s[0] == 'Railways':
	# 				to_nodes_df = to_nodes_df[to_nodes_df['objectid'] >  0]

	# 			sindex_to_nodes = to_nodes_df.sindex

	# 			from_nodes_df.rename(columns={'node_id': 'from_node','geometry':'from_geom'}, inplace=True)
	# 			# from_nodes_df.rename(columns={'node_id': 'from_node'}, inplace=True)

	# 			# print (from_nodes_df)
	# 			# print (from_nodes_df)

	# 			from_nodes_df['from_mode'] = f_s[0].lower().strip()
	# 			from_nodes_df['to_mode'] = t_s[0].lower().strip()


	# 			# if t_s[0] == 'Roads':
	# 			# 	to_edges_df = gpd.read_file(to_edges_in)
	# 			# 	to_edges_df.columns = map(str.lower, to_edges_df.columns)
	# 			# 	sindex_to_edges = to_edges_df.sindex
	# 			# 	from_nodes_df['edge_prox'] = from_nodes_df.from_geom.apply(lambda x: get_nearest_node_on_line(x,to_edges_df,'edge_id'))
	# 			# 	from_nodes_df[['to_node','to_geom']] = from_nodes_df['edge_prox'].apply(pd.Series)
	# 			# 	from_nodes_df.drop('edge_prox',axis=1,inplace=True)


	# 			# else:
	# 			# 	from_nodes_df['to_node'] = from_nodes_df.from_geom.apply(lambda x: get_nearest_node(x,sindex_to_nodes,to_nodes_df,'node_id'))
	# 			# 	from_nodes_df['to_geom'] = from_nodes_df.from_geom.apply(lambda x: get_nearest_node(x,sindex_to_nodes,to_nodes_df,'geometry'))

	# 			from_nodes_df['to_node'] = from_nodes_df.from_geom.apply(lambda x: get_nearest_node(x,sindex_to_nodes,to_nodes_df,'node_id'))
	# 			from_nodes_df['to_geom'] = from_nodes_df.from_geom.apply(lambda x: get_nearest_node(x,sindex_to_nodes,to_nodes_df,'geometry'))
	# 			print (from_nodes_df)
	# 			from_nodes_df['geometry'] = from_nodes_df.apply(lambda x: LineString([x.from_geom,x.to_geom]),axis = 1)

	# 			all_multi_pts += list(zip(from_nodes_df['from_node'].values.tolist(),from_nodes_df['from_mode'].values.tolist(),from_nodes_df['from_geom'].values.tolist())) + list(zip(from_nodes_df['to_node'].values.tolist(),from_nodes_df['to_mode'].values.tolist(),from_nodes_df['to_geom'].values.tolist()))
	# 			from_nodes_df.drop('from_geom',axis=1,inplace=True)
	# 			from_nodes_df.drop('to_geom',axis=1,inplace=True)

	# 			# from_nodes_df = from_nodes_df.rename(columns={'line_geometry':'geometry'}).set_geometry('geometry')
	# 			# from_nodes_df['g_id'] = from_nodes_df.index.values.tolist()
	# 			# from_nodes_df['edge_id'] = ['multie_{}'.format(x) for x in from_nodes_df['g_id'].values.tolist()]
	# 			if f_s[0] == 'Waterways':
	# 				from_nodes_df = from_nodes_df[['from_node','to_node','from_mode','to_mode','port_type','port_class','geometry']]
	# 			else:
	# 				from_nodes_df = from_nodes_df[['from_node','to_node','from_mode','to_mode','geometry']]
	# 			all_multi.append(from_nodes_df)
	# 			# from_nodes_df = GeoDataFrame(from_nodes_df, geometry='geometry',crs='epsg:4326')
	# 			# from_nodes_df = from_nodes_df.to_crs({'init': 'epsg:4326'})
	# 			# out_shp = os.path.join(data_path,'Multi')
	# 			# from_nodes_df.to_file()

	# all_multi = gpd.GeoDataFrame(pd.concat(all_multi, axis=0, sort = 'False', ignore_index=True).fillna('none'),crs = 'epsg:4326')

	# all_multi['g_id'] = all_multi.index.values.tolist()
	# all_multi['edge_id'] = ['multie_{}'.format(x) for x in all_multi['g_id'].values.tolist()]
	# all_multi['length'] = all_multi.geometry.apply(line_length)
	# all_multi = all_multi[['edge_id','g_id','from_node','to_node','from_mode','to_mode','port_type','port_class','length','geometry']]
	# out_shp = os.path.join(data_path,'Multi','multi_edges','multimodal_edges.shp')
	# all_multi.to_file(out_shp)

	# cols = ['node_id','mode','geometry']
	# all_multi_pts_df = gpd.GeoDataFrame(pd.DataFrame(all_multi_pts,columns = cols),crs = 'epsg:4326')
	# out_shp = os.path.join(data_path,'Multi','multi_edges','multimodal_nodes.shp')
	# all_multi_pts_df.to_file(out_shp)

	'''
	+++++++++++++++++++++++++++++++++++++++++++++++++
	create multi-modal edges
	+++++++++++++++++++++++++++++++++++++++++++++++++
	'''
	'''
	Elco IO numbers - negative implies loss
	IO Folder - 6_mria
	'''

	'''
	++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	modify existing road network with introduction of new nodes
	++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	'''

	# from_sector = [('Waterways','waterways'),('Railways','national_rail')]
	# to_sector = ('Roads','national_roads')

	# all_multi = []
	# mode_data_path = os.path.join(data_path,'Roads','national_roads')
	# for file in os.listdir(mode_data_path):
	# 	try:
	# 		if file.endswith(".shp") and 'edges' in file.lower().strip():
	# 			to_edges_in = os.path.join(mode_data_path, file)

	# 		if file.endswith(".shp") and 'nodes' in file.lower().strip():
	# 			to_nodes_in = os.path.join(mode_data_path, file)
	# 	except:
	# 		print ('Road nodes and edge files necessary')

	# to_nodes_df = gpd.read_file(to_nodes_in)
	# to_nodes_df.columns = map(str.lower, to_nodes_df.columns)

	# to_edges_df = gpd.read_file(to_edges_in)
	# to_edges_df.columns = map(str.lower, to_edges_df.columns)
	# sindex_to_edges = to_edges_df.sindex

	# for f_s in from_sector:
	# 	mode_data_path = os.path.join(data_path,f_s[0],f_s[1])
	# 	for file in os.listdir(mode_data_path):
	# 		try:
	# 			if file.endswith(".shp") and 'nodes' in file.lower().strip():
	# 				from_nodes_in = os.path.join(mode_data_path, file)
	# 		except:
	# 			print ('Network nodes and edge files necessary')

	# 	from_nodes_df = gpd.read_file(from_nodes_in).fillna(0)
	# 	from_nodes_df.columns = map(str.lower, from_nodes_df.columns)
	# 	if f_s[0] == 'Waterways':
	# 		from_nodes_df = from_nodes_df[(from_nodes_df['port_type'] == 'inland') | (from_nodes_df['port_type'] == 'sea')]
	# 	elif f_s[0] == 'Railways':
	# 		from_nodes_df = from_nodes_df[from_nodes_df['objectid'] >  0]

	# 	# print (from_nodes_df)

	# 	from_nodes_df['edge_prox'] = from_nodes_df.geometry.apply(lambda x: get_nearest_node_on_line(x,to_edges_df,'edge_id'))
	# 	from_nodes_df[['to_edge','to_geom']] = from_nodes_df['edge_prox'].apply(pd.Series)
	# 	from_nodes_df.drop('edge_prox',axis=1,inplace=True)
	# 	from_nodes_df['line_geometry'] = from_nodes_df.apply(lambda x: LineString([x.geometry,x.to_geom]),axis = 1)
	# 	from_nodes_df['length'] = from_nodes_df.line_geometry.apply(line_length)

	# 	prox_df = from_nodes_df[['node_id','to_edge','to_geom','length']]
	# 	prox_df.rename(columns={'to_edge':'edge_id','to_geom': 'geometry'}, inplace=True)

	# 	all_multi.append(prox_df)

	# all_multi = gpd.GeoDataFrame(pd.concat(all_multi, axis=0, sort = 'False', ignore_index=True).fillna('none'),crs = 'epsg:4326')
	# print (all_multi)

	# length_thr = 3.0
	# all_multi = all_multi[all_multi['length'] <= length_thr]
	# modify_network_with_new_points(all_multi,to_edges_df,to_nodes_df,'edge_id','node_id')

	'''
	++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	modify existing road network with introduction of new nodes
	++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	'''

	'''
	Clean road edge network
	'''

	# edges = gpd.read_file(os.path.join(data_path,'Roads','national_roads','national_network_edges.shp'))
	# edges.columns = map(str.lower, edges.columns)
	# edge_columns = edges.columns.values.tolist()
	# start_cols = ['edge_id','g_id','from_node','to_node']
	# rest_cols = [c for c in edge_columns if c not in start_cols]
	# cols = start_cols + rest_cols

	# edges_in = edges[cols]
	# edges_in.to_file(os.path.join(data_path,'Roads','national_roads','national_network_edges.shp'))
