"""Summarise hazard data


Susceptibility levels
---------------------
5 - rất cao - very high
4 - cao - high
3 - trung bình - medium
2 - thấp - low
1 - rất thấp - very low

Filename abbreviations
----------------------
lsz_45_2025 - landslide susceptibility zones under RCP 4.5 in 2025
lsz_85_2025 - landslide susceptibility zones under RCP 8.5 in 2025
ffsz_rcp45_25 - flash flood susceptibility zones under RCP 4.5 in 2025
ffsz_rcp85_25 - flash flood susceptibility zones under RCP 8.5 in 2025

Format notes
------------
MapInfo files
 - .TAB is main file, .DAT, .ID, .MAP typically go alongside (.WOR is a workspace)
 - .TAB/.txt or .TAB/.tif pairs for raster data

ArcGrid files
 - hdr.adf is main file

ESRI Shapefiles
 - .shp is main file

"""
import configparser
import glob
import os

import fiona


def main():
    config_path = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'config.ini'
        )
    )
    config = configparser.ConfigParser()
    config.read(config_path)
    incoming_data_path = config['paths']['incoming_data']

    hazard_path = os.path.join(incoming_data_path, 'Natural_Hazard_Maps', 'Maps')
    exts = ['shp', 'TAB', 'adf']
    for file_path in glob.glob(str(hazard_path + '**/*.{shp,adf,TAB,tif}')):
        get_info(file_path)

def get_info(file_path):
    print(file_path)
    with fiona.open(file_path, 'r') as source:
        print(source.schema)


if __name__ == '__main__':
    main()
