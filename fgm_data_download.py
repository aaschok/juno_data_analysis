#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:01:33 2021

@author: aschok

Downloads magnitometer data from Nasa's pds archive between two dates into csv files
"""

import datetime
import re
import requests
import pandas as pd

start_date = datetime.datetime.strptime('2016-267', '%Y-%j')
end_date = datetime.datetime.strptime('2016-291', '%Y-%j')
date_range = pd.date_range(start_date.isoformat(), end_date.isoformat(), freq='D')
fgm_data_index_df = pd.read_csv('/data/juno_spacecraft/data/index/fgm_data_index.TAB')
cols = ['VOLUME_ID  ', 'SID             ', 'DATA_SET_ID         ',
        'PRODUCT_ID                    ', 'START_TIME         ',
        'STOP_TIME          ',
        'FILE_SPECIFICATION_NAME                                       ',
        'CR_DATE        ', 'PRODUCT_LABEL_MD5CHECKSUM       ']
file_name_regex = re.compile(r'fgm(.+?)$')
date_regex = re.compile(r'\d{7}')
for index, row in fgm_data_index_df.iterrows():
    if row[cols[1]].strip(' ') == 'PC 1 SECOND':
        file_date = datetime.datetime.strptime(date_regex.search(row[6]).group(), '%Y%j')
        if file_date in date_range:
            archive_file_loc = row[6].strip(' ').strip('.lbl')
            file_name = file_name_regex.search(archive_file_loc).group()
            r = requests.get(f'https://pds-ppi.igpp.ucla.edu/ditdos/write?id=pds://PPI/JNO-J-3-FGM-CAL-V1.0/{archive_file_loc}&f=csv')
            file = open(f'/home/aschok/Documents/data/{file_name}.csv', 'w')
            file.write(r.text)
            file.close()
            print(f'Saved {file_name}')
