# Scrape HTML details from VIWA geoserver

# URL template: http://hatang.viwa.gov.vn/BanDo/_ChiTietCangBen?id=530

import os
import sys
import re
import pandas as pd
import geopandas as gpd
import shapely

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.utils import load_config

def check_empty_geom(geom):
	if geom:
		return geom.wkb
	else:
		return ''

def main():
	data_path,calc_path,output_path = load_config()['paths']['data'],load_config()['paths']['calc'],load_config()['paths']['output']
	root_dir = os.path.join(data_path,'Waterways','viwa','html')
	r = re.compile('LuongHangHoaThongQua">([^<]*)')

	viwa_ports_tons = []
	for root, dirs, files in os.walk(root_dir):
		for file in files:
			if file.endswith(".html") and 'detail' not in file:
				input_file = os.path.join(root, file)
				port_no = re.findall('\d+', file)
				with open(input_file, 'r') as fh:
					for line in fh:
						m = r.search(line)
						if m:
							if m.group(1).isdigit():
								viwa_ports_tons.append((int(port_no[0]),float(m.group(1))))
							else:
								viwa_ports_tons.append((int(port_no[0]),0))

	viwa_ports_tons_df = pd.DataFrame(viwa_ports_tons,columns = ['cangbenid','tons'])
	viwa_ports_tons_df = viwa_ports_tons_df[viwa_ports_tons_df['tons'] > 0]
	print(len(viwa_ports_tons_df.index))
	print (viwa_ports_tons_df['cangbenid'].values.tolist())

	shp_file = os.path.join(data_path,'Waterways','viwa','cang-ports.shp')
	ports = gpd.read_file(shp_file)
	ports.columns = map(str.lower, ports.columns)
	cols = ports.columns.values.tolist()
	print (len(ports.index))
	# print (ports['cangbenid'].values.tolist())
	ports_with_geom = []
	id_list = []
	for iter_,values in ports.iterrows():
		if values['geometry'] and values['cangbenid'] not in id_list:
			if int(values['cangbenid']) in viwa_ports_tons_df['cangbenid'].values.tolist():
				print ('match')
				tons = viwa_ports_tons_df.loc[viwa_ports_tons_df['cangbenid'] == int(values['cangbenid']),'tons'].sum()
				ports_with_geom.append(tuple(list(values) + [tons]))
			else:
				ports_with_geom.append(tuple(list(values) + [0]))
			
			id_list.append(values['cangbenid'])

	ports_df = pd.DataFrame(ports_with_geom,columns = cols + ['tons'])
	ports = gpd.GeoDataFrame(ports_df,crs='epsg:4326')
	# print (len(ports.index))
	print (ports)

	ports.to_file(os.path.join(data_path,'Waterways','viwa_select','iwt_ports.shp'))

	# ports = ports[ports['geometry'] != None]
	# ports = ports.to_crs({'init': 'epsg:4326'})
	# ports.columns = map(str.lower, ports.columns)
	# print (ports.columns.values.tolist())
	# print (len(ports.index))
	# print (ports)

	# convert to wkb
	# ports["geometry"] = ports["geometry"].apply(lambda geom: geom.wkt)
	# print (ports)
	# cols = ports.columns.values.tolist()
	# ports = ports.drop_duplicates(subset=cols, keep=False)
	# # # convert back to shapely geometry
	# ports["geometry"] = ports["geometry"].apply(lambda geom: shapely.wkt.loads(geom))
	# # print (len(ports.index))
	# print (ports)
	# ports = ports[ports['geometry'] != '']
	# print (len(ports.index))


if __name__ == '__main__':
	main()
