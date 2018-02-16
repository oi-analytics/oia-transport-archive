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
import csv
import glob
import os

import fiona
import fiona.crs
import rasterio


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
    vector_exts = ['.shp', '.TAB']
    raster_exts = ['.tif', '.txt']
    pattern = os.path.join(hazard_path, '**', '*.*')

    report_path = os.path.join(hazard_path, 'report.csv')

    with open(report_path, 'w', newline='') as output_fh:
        writer = csv.writer(output_fh)
        writer.writerow(('filename', 'path', 'format', 'type', 'crs', 'bounds', 'number_of_features', 'fields'))

        for file_path in glob.glob(pattern, recursive=True):
            ext = os.path.splitext(file_path)[1]
            filename = os.path.split(file_path)[1]
            file_path_detail = str(file_path).replace(
                str(incoming_data_path),
                ''
            )
            if ext in vector_exts:
                details = vector_details(file_path)
                if details:
                    fields, geometry_type, crs, bounds, number_of_features = details
                    row = (
                        filename,
                        file_path_detail,
                        ext,
                        'vector:{}'.format(geometry_type),
                        crs,
                        bounds,
                        number_of_features,
                        fields
                    )
                    writer.writerow(row)

            if ext in raster_exts or filename == 'hdr.adf':
                bands, crs, bounds, number_of_cells = raster_details(file_path)
                row = (
                    filename,
                    file_path_detail,
                    ext,
                    'raster',
                    crs,
                    bounds,
                    number_of_cells,
                    bands
                )
                writer.writerow(row)

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

if __name__ == '__main__':
    main()
