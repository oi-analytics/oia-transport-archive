
# -*- coding: utf-8 -*-
"""
Python script to assign commodity flows on the road network
Created on Wed Nov 26 2017

@author: Raghav Pant
"""

import pandas as pd
import os
import psycopg2
import networkx as nx
import csv
from sqlalchemy import create_engine
import subprocess as sp

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


from vtra.utils import load_config

def vector_details(file_path):
	try:
		with fiona.open(file_path, 'r') as source:
			fields = [(k,v) for k, v in source.schema['properties'].items()]
			geometry_type = source.schema['geometry']
			crs = fiona.crs.to_string(source.crs)
			bounds = source.bounds
			number_of_features = len(source)
		return fields, geometry_type, crs, bounds,number_of_features
	except Exception as ex:
		print("INFO: fiona read failure (likely not a vector file):", ex)
		return None

def raster_details(file_path):
	with rasterio.open(file_path) as dataset:
		bbox = dataset.bounds
		bounds = (bbox.left, bbox.bottom, bbox.right, bbox.top)
		number_of_cells = dataset.width * dataset.height
		if dataset.crs.is_valid:
			crs = dataset.crs.to_string()
		else:
			crs = 'invalid/unknown'
		bands = [(i, dtype) for i, dtype in zip(dataset.indexes, dataset.dtypes)]
	return bands, crs, bounds, number_of_cells

def add_column_to_table(table_name, table_match, col_name, col_id,col_type, cursor, connection):
	sql_query = "alter table %s add column %s %s"%(table_name,col_name,col_type)
	cursor.execute(sql_query)
	connection.commit()


	# sql_query = '''
	# 			update %s set %s = (select %s from %s as A where %s.%s = A.%s)
	# 			'''%(table_name,col_name,col_name,table_match,table_name,col_id,col_id)

	sql_query = '''
				UPDATE {0} t2 SET {1} = t1.{2}
				FROM {3} t1 WHERE  t2.{4} = t1.{5}
				'''.format(table_name,col_name,col_name,table_match,col_id,col_id)
	cursor.execute(sql_query)
	connection.commit()

	sql_query = "update %s set %s = 0 where %s is Null"%(table_name,col_name,col_name)
	cursor.execute(sql_query)
	connection.commit()

def main():
	curdir = os.getcwd()
	conf = load_config()

	try:
		conn = psycopg2.connect(**conf['database'])
	except:
		print ("I am unable to connect to the database")


	cur = conn.cursor()

	engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format({
		**conf['database']
	}))

	shp_file_path = os.path.join(conf['paths']['data'], 'Hazards_data', 'coastal_typhoon_floods')

	'''
	Get the road flooding results
	'''
	layers_ids = [('roads_edge_flood','edge_id','road2009edges','geom'),
				('rail_edge_flood','edge_id','railnetworkedges','geom'),
				('roads_node_flood','node_id','road2009nodes','geom'),
				('rail_node_flood','node_id','railnetworknodes','geom'),
				('port_flood','node_id','seaport_nodes','geom'),
				('airport_flood','node_id','airport_nodes','geom'),
				('provinces_flooding','name_eng','province_level_stats','geom')]

	flood_list = []
	excel_writer = pd.ExcelWriter('vnm_flood_results.xlsx')
	for l in layers_ids:
		sql_query = "drop table if exists %s"%(l[0])
		cur.execute(sql_query)
		conn.commit()

		sql_query = "create table %s as select %s,%s from %s"%(l[0],l[1],l[3],l[2])
		cur.execute(sql_query)
		conn.commit()

		for file in os.listdir(shp_file_path):
			if file.endswith(".shp") and 'mask_1' in file:
				# print (file)
				# print(os.path.join(shp_file_path, file))

				fpath = os.path.join(shp_file_path, file)
				flds, geom_type, cr, bds,n_f = vector_details(fpath)

				fname = file.split(".shp")
				fname = fname[0]
				flood_list.append(fname)
				cr_split = cr.split(':')
				print (cr_split)
				fr_srid = [c for c in cr_split if c.isdigit() is True]
				if fr_srid:
					print (fr_srid)
				elif 'wgs84' in cr.lower().strip():
					fr_srid = '4326'

				command = "shp2pgsql -I -s {0}:4326 -d {1} {2} | psql -U {3} -d {4}".format(fr_srid,fpath,fname, conf['database']['user'], conf['database']['database'])
				# sp.call(command.split())
				sp.call(command, shell=True)


				sql_query = "drop table if exists dummy_flood"
				cur.execute(sql_query)
				conn.commit()

				if l[0] in ['roads_edge_flood','rail_edge_flood']:
					sql_query = '''create table dummy_flood as select A.%s,
								sum(st_length(st_intersection(A.%s::geography,B.geom::geography)))/1000 as %s
								from %s as A, %s as B where st_intersects(A.%s,B.geom) is True and st_isvalid(A.%s) and st_isvalid(B.geom)
								group by A.%s'''%(l[1],l[3],fname,l[2],fname,l[3],l[3],l[1])

					cl_type = 'double precision'

				elif l[0] in ['roads_node_flood','rail_node_flood','port_flood','airport_flood']:
					sql_query = '''create table dummy_flood as select A.%s, 1 as %s
								from %s as A, %s as B where st_within(A.%s,B.geom) is True
								'''%(l[1],fname,l[2],fname,l[3])
					cl_type = 'integer'

				else:
					sql_query = '''create table dummy_flood as select A.%s,
								sum(st_area(st_intersection(A.%s::geography,B.geom::geography)))/1000000 as %s
								from %s as A, %s as B where st_intersects(A.%s,B.geom) is True and st_isvalid(A.%s) and st_isvalid(B.geom)
								group by A.%s'''%(l[1],l[3],fname,l[2],fname,l[3],l[3],l[1])

					cl_type = 'double precision'


				cur.execute(sql_query)
				conn.commit()
				add_column_to_table(l[0],'dummy_flood', fname, l[1],cl_type, cur, conn)

		'''
		Write everthing to a pandas dataframe
		'''
		df = pd.read_sql_table(l[0], engine)
		df.to_excel(excel_writer,l[0], index = False)
		excel_writer.save()


	'''
	Do some diagnostics
	'''
	cur.close()
	conn.close()

if __name__ == '__main__':
	main()
