# Scrape HTML details from VIWA geoserver

# URL template: http://hatang.viwa.gov.vn/BanDo/_ChiTietCangBen?id=530

import os
import json

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils import load_config

def main(input_file, output_dir):
	with open(input_file, 'r') as fh:
		ports = json.load(fh)

	for port in ports:
		port_id = port['CangBenID']

		output_path = os.path.join(output_dir, 'cangben-{}.html'.format(port_id))
		if not os.path.exists(output_path):
			print(port_id)
			try:	
				r = requests.get(
					'http://hatang.viwa.gov.vn/BanDo/_ChiTietCangBen',
					params={
						'id': port_id
					}
				)
				with open(output_path, 'w', encoding='utf-8') as fh:
					fh.write(r.text)
			except requests.exceptions.ConnectionError:
				print("Error", port_id)

		# also request details
		output_path = os.path.join(output_dir, 'cangben-{}-detail.html'.format(port_id))
		if not os.path.exists(output_path):
			print(port_id, "details")
			try:
				r = requests.get(
					'http://cangben.viwa.gov.vn/trangchu/cangben/{}'.format(port_id)
				)
				with open(output_path, 'w', encoding='utf-8') as fh:
					fh.write(r.text)
			except requests.exceptions.ConnectionError:
				print("Error", port_id)


if __name__ == '__main__':
	conf = load_config()
	PORT_LIST_PATH = os.path.join(conf['paths']['data'], 'Waterways', 'viwa', 'datacangben-all.json')
	OUTPUT_DIR = os.path.join(conf['paths']['data'], 'Waterways', 'viwa', 'html')
	main(PORT_LIST_PATH, OUTPUT_DIR)
