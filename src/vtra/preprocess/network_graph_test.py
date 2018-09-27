"""
Get vietnam shapefiles and see their network graphs
@author: Raghav Pant
Date: July 15, 2018
REMOVE THIS FROM THE GITHUB LATER AFTER TESTING
"""
import os
import sys
import igraph as ig

import vtra.preprocess.network_create as nc
import vtra.transport_network_creation as tnc
from vtra.utils import *

def main():
	config = load_config()
	input_path = os.path.join(config['paths']['data'],'Roads','laocai_roads','vietbando_laocai_edges.shp')
	net = tnc.shapefile_to_network(input_path)
	net = tnc.add_igraph_costs_province_roads(net,0.019)



if __name__ == '__main__':
	main()
