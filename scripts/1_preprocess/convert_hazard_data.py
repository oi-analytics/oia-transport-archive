# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 08:19:55 2018

@author: elcok
"""

import os
import subprocess
import json

def main():
    # =============================================================================
    #     # Define current directory and data directory
    # =============================================================================
    config_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    )
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    data_path = config['paths']['data']
    calc_path = config['paths']['calc']
    
    # =============================================================================
    #     # convert hazard data to ascii if that is not done yet
    # =============================================================================
    raw_flood_data = os.path.join(data_path,'Floods','Thanh_Hoa','MapInfo_Data')
    
    MapInfo_river_floods_TH = [os.path.join(raw_flood_data,x) for x in os.listdir(raw_flood_data) if x.endswith('.txt')]
    tiff_river_floods_TH = [os.path.join(data_path,'Floods','Thanh_Hoa','Flood_Maps',x)[:-3]+'tiff' for x in os.listdir(raw_flood_data) if x.endswith('.txt')]

    x = 0
    for raw_river_flood in MapInfo_river_floods_TH:

        # convert maps to geotiff for more easy processing
        os.system('gdalwarp -overwrite %s %s -s_srs EPSG:32648 -t_srs EPSG:4326 -co compress=LZW' % (raw_river_flood,tiff_river_floods_TH[x]))
   
        tmpfile_1 = os.path.join(calc_path,'tmpfile_1.tiff')
        outfile = os.path.join(tiff_river_floods_TH[x][:-5]+'_1m_thres.shp')

        #polygonize the rasters
        convert(1,tiff_river_floods_TH[x],tmpfile_1,outfile)
        
        x += 1

def convert(threshold, infile, tmpfile_1, outfile):
    """Threshold raster, convert to polygons
    """
    args = [
        "gdal_calc.py",
        '-A', infile,
        '--outfile={}'.format(tmpfile_1),
        '--calc=logical_and(A>={}, A<999)'.format(threshold),
        '--type=Byte', '--NoDataValue=0',
        '--co=SPARSE_OK=YES',
        '--co=NBITS=1',
        '--co=COMPRESS=LZW'
    ]
    subprocess.run(args)

    subprocess.run([
        "gdal_polygonize.py",
        tmpfile_1,
        '-q',
        '-f', 'ESRI Shapefile',
        outfile
    ])

if __name__ == "__main__":
    main()

