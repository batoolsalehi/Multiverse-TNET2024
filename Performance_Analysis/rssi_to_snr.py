########################################################
##### Project: digital twin FLASH
##### Contact: Batool Salehi
##### The script adds the SNR to the csv files in a new column
##### Change line 71 to generate the updated csv files for each scenario
########################################################

import os,glob
import pandas as pd
import numpy as np
import csv
import time
import datetime
import collections
from tqdm import tqdm

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


def extract_sweepfile(text_file):
    ###gets text files from sweeping and returns a dictionary with {(sector,rssi):snr}
    sweep_file = open(text_file, "r")
    sweep_lines = sweep_file.readlines()
    end_of_sweep_flag = '---------------End of One Sector Sweeping--------------'
    sweep_count = 0
    sector_rssi_snr = {}   #(sector,rssi):snr

    sector_list = []
    rssi_list = []
    snr_list = []
    for line in sweep_lines:
        if 'sec:' in line:
            sector_list.append(int(line[7:10].strip()))
            rssi_list.append(int(line[16:25].strip()))
            # print("line[35:38]",line[39:42])
            snr_list.append(int(line[39:42].strip()))
        if line.strip() == end_of_sweep_flag:
            sector_rssi_snr[sweep_count] = (sector_list,rssi_list,snr_list)
            sector_list = []
            rssi_list = []
            snr_list = []
            sweep_count+=1

    return sector_rssi_snr


def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    if extension not in input_path:
        files_list=[]
        for path, subdirs, files in os.walk(input_path):
            for file in files:
                if file.endswith(extension):
                   files_list.append(os.path.join(path, file))
        return files_list
    else:
        return [input_path]

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)


main_directory = '/Users/maryam/Desktop/digital_twin_flash/flash/'
all_files = show_all_files_in_directory(main_directory,'Synchornized_data.csv')
print("total of files detected:",len(all_files))


for files in all_files:
    print("files",files)
    if "opposite" and "Cat3" in files.split('/'):
        RF_file = main_directory+'/GT/GT/Sweep_'+files.split('/')[-3]+'_'+files.split('/')[-2].split('_')[-1]+'.txt'
        extract_sweepfile_dictionary = extract_sweepfile(RF_file)
        with open(files, 'r', encoding='UTF8', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        data_snr = []
        c = 0
        for d in data[1:]:   #excluding header and sweeping all elements in current csv file in flash
            sectors_csv = d[16]
            rssi_csv = d[15]
            c+=1
            ####compare with extract_sweepfile_dictionary
            for k in extract_sweepfile_dictionary.keys():
                if str(extract_sweepfile_dictionary[k][0])==sectors_csv and str(extract_sweepfile_dictionary[k][1])==rssi_csv:
                    print("extract_sweepfile_dictionary[k][2]",extract_sweepfile_dictionary[k][2])
                    data_snr.append(d+[extract_sweepfile_dictionary[k][2]])
                    break
            print("len(data_snr)",len(data_snr),c)

        header = data[0]+['snr']

        with open(files[:-4] +'_with_SNR.csv', 'w', encoding='UTF8', newline='') as f:    ### last two alphabets are GT in all directories
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_snr)
