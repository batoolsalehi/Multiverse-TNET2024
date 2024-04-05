########################################################
##### Project: digital twin FLASH
##### Contact: Batool Salehi
##### Generate labels from WI for augmentation part
##### Change inputs line 81-87 for each setting. Outliear is set as False here to generate the same number of samples as FLASH.
########################################################
from __future__ import division

import os
import csv
import argparse
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import ast
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import Counter
from scipy import stats
from sklearn.metrics import ndcg_score
import random
import math
from collections import Counter
import scipy.io
import pickle
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (6,4)
from math import radians, cos, sin, asin, sqrt

def distance(lat1, lat2, lon1, lon2):

    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    return(c * r)


def wi_arrange(sector_power_list):
    valid_sectors = [x for x in range(1,32)]+[61,62,63]
    s = []
    p = []
    for v in valid_sectors:
        for k in sector_power_list:
            # print("k",k)
            if float(k[0].split("_")[-1]) == v:
                s.append(int(k[0].split("_")[-1]))
                p.append(k[1])
    return s,p

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

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

####Inputs
WI_coordinates_file = "/Users/maryam/Desktop/digital_twin_flash/Batool/WI_coordinates-oppositeLane.csv"
WI_powers_file = "/Users/maryam/Desktop/digital_twin_flash/Batool/Twin-2_NLOS.csv"
# flash = "/Users/maryam/Desktop/digital_twin_flash/flash/Cat1/opposite/Extract/"
flash = "/Users/maryam/Desktop/digital_twin_flash/flash/Cat3/static_in_front/Extract/"
open_area = "/Users/maryam/Desktop/digital_twin_flash/Batool/Twin-1_NLOS.csv"
mat_file = "/Users/maryam/Desktop/digital_twin_flash/talon-sector-patterns-master/legacy_measurements/sectorpattern_3d_sta.mat"

ouliear_remove = False
arranging = False  ##for best results outliear True and arrangin false
distance_limit_tx = 0.05
thershold =0
correlation_window = 30
tx_lat, tx_long = 42.3378046793832, -71.08663802513
twin = "twin1"

######################################################
########################WI##############################
######################################################
#####arrange coordinates and powers of WI into dictionary
if twin=="twin3" or twin=="twin2":
    print("#######twin2 or twin3########")
    ####extract coordinates
    with open(WI_coordinates_file, 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        data_coordinates = list(reader)
    WI_coordinates = []
    for d in data_coordinates[2:]:
        WI_coordinates.append((float(d[2]),-float(d[3])))
    ####extract powers
    WI_beam_powers = []
    df = pd.read_csv(WI_powers_file)
    for l in range(1,201):
        beam_power = []
        antenna_column = df['Antennas-'+str(l)]
        power_column = df['Power-'+str(l)+' (dBm)']
        for a in range(len(antenna_column)):
            beam_power.append((antenna_column[a],power_column[a]))
        WI_beam_powers.append(beam_power)
    ###combine coordinates and powers, structure   {coord:['legacy1':0.12,..,'legacy14':0.9]}
    coordinates_beam_powers = {}
    for l in range(len(WI_beam_powers)):
        coordinates_beam_powers[WI_coordinates[l]] = WI_beam_powers[l]
elif twin=="twin1":
    print("#######twin1########")
    secotrs_mat_file = scipy.io.loadmat(mat_file)['pattern_snr'][0]  ###only elevation for now  (101,34)
    print("secotrs_mat_file",secotrs_mat_file,secotrs_mat_file.shape,secotrs_mat_file[97])
    ####extract coordinates
    with open(WI_coordinates_file, 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        data_coordinates = list(reader)
    WI_coordinates = []
    for d in data_coordinates[2:]:
        WI_coordinates.append((float(d[2]),-float(d[3])))
    print("WI_coordinates",len(WI_coordinates))
    ####extract powers
    df = pd.read_csv(open_area)
    beam_power = {}
    antenna_column = df['Antenna']
    power_column = df['Max Propogated Power(dBm)']
    for a in range(len(antenna_column)):
        beam_power[antenna_column[a]] = power_column[a]
    # print("beam_power",beam_power)

    WI_beam_powers = []
    for l in range(0,200):
        valid_sectors = [x for x in range(1,32)]+[61,62,63]
        beam_power_location = []
        for a in range(34):
            if l<=100:
                angle_tx_rx = getAngle((WI_coordinates[l][0], WI_coordinates[l][1]),(tx_lat, tx_long), (WI_coordinates[100][0], WI_coordinates[100][1]))
                look_up_angle = 100+angle_tx_rx
            else:
                angle_tx_rx = getAngle((WI_coordinates[100][0], WI_coordinates[100][1]),(tx_lat, tx_long),(WI_coordinates[l][0], WI_coordinates[l][1]))
                look_up_angle = 100-angle_tx_rx
            # print("angle_tx_rx",angle_tx_rx)
            index_to_look = int(look_up_angle/2) if int(look_up_angle/2)<=100 else 100
            if math.isnan(secotrs_mat_file[index_to_look][a]*beam_power['Legacy_'+str(valid_sectors[a])]):
                beam_power_location.append(('Legacy_'+str(valid_sectors[a]),0))
            else:
                beam_power_location.append(('Legacy_'+str(valid_sectors[a]),secotrs_mat_file[index_to_look][a]*beam_power['Legacy_'+str(valid_sectors[a])]))
        # print("look_up_angle",l,look_up_angle)
        WI_beam_powers.append(beam_power_location)
    ###combine coordinates and powers, structure   {coord:['legacy1':0.12,..,'legacy14':0.9]}
    coordinates_beam_powers = {}
    for l in range(len(WI_beam_powers)):
        coordinates_beam_powers[WI_coordinates[l]] = WI_beam_powers[l]


# print("coordinates_beam_powers",coordinates_beam_powers)
######################################################
########################FLASH##############################
######################################################
##########read points from FLASH
all_csv_flash = show_all_files_in_directory(flash,'Synchornized_data_with_SNR.csv')
print("Total of %s files detected in FLASH" % len(all_csv_flash))
probabilities = {}
total_samples , ouliers =0,0
for flash_csv in all_csv_flash:
    print("**************************************************",flash_csv)
    with open(flash_csv, 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        data_flash = list(reader)
    total_samples+=len(data_flash)
    ###outliear
    if ouliear_remove is True:
        data_flash_without_outliear = []
        for d in range(1,len(data_flash)):
            if distance(float(data_flash[d][8]),tx_lat,float(data_flash[d][9]),tx_long)<distance_limit_tx:   ###remove outliers
                data_flash_without_outliear.append(data_flash[d])
            else:
                ouliers+=1
        data_flash_without_outliear.insert(0,data_flash[0])  ##adding the header to keep the structure
    else:
        data_flash_without_outliear = data_flash

    ###arranging
    if arranging is True:
        distance_from_tx_data = {}
        origin_lat, origin_long = float(data_flash_without_outliear[1][8]) , float(data_flash_without_outliear[1][9])
        for d in range(1,len(data_flash_without_outliear)):
            lat_flash = float(data_flash_without_outliear[d][8])
            long_flash = float(data_flash_without_outliear[d][9])
            distance_from_tx_data[d] = distance(lat_flash,origin_lat,long_flash,origin_long)
        all_distances = list(distance_from_tx_data.values())
        indexes_decreasing = sorted(range(len(all_distances)), key=lambda k: all_distances[k],reverse=False)
        data_flash_arranged = []
        for i in indexes_decreasing:
            data_flash_arranged.append(data_flash_without_outliear[i+1])
        data_flash_arranged.insert(0,data_flash_without_outliear[0])  ##adding the header to keep the structure
    else:
        data_flash_arranged = data_flash_without_outliear

    print("compare lens",len(data_flash),len(data_flash_without_outliear),len(data_flash_arranged))


    ####correlation
    previous_lat = 0
    previous_long = 0
    startpoint_probaility = {}
    for start_point in range(correlation_window):
        start_point_inner = start_point
        failuar = 0
        counter = 0
        for d in data_flash_arranged[1:]:
            found = False
            lat_flash = float(d[8])
            long_flash = float(d[9])
            rssi_flash = ast.literal_eval(d[15])
            snr_flash = ast.literal_eval(d[19])
            ids_flash = ast.literal_eval(d[16])

            #####remove dplicates of rssi/power measurments
            counting = Counter(ids_flash)
            unique_ids = list(counting.keys())
            unique_snr = []
            unique_rssi = []
            for u in unique_ids:
                repeats = [i for i in range(len(ids_flash)) if ids_flash[i]==u]
                ###only keep maximum
                unique_snr.append(max([snr_flash[r] for r in repeats]))
                unique_rssi.append(max([rssi_flash[r] for r in repeats]))

            ###determin the flash sectors within thershold
            within_thershold_range = []
            for f in range(len(unique_snr)):
                if unique_snr[f]>=unique_snr[unique_rssi.index(max(unique_rssi))]-thershold:    ###snr of the best sector in flash minues thershold
                    within_thershold_range.append(unique_ids[f])

            #####mapping
            if previous_long==0:
                previous_long = long_flash
                previous_lat = lat_flash
                distance_cm = 0
            else:
                distance_cm = 100000*distance(lat_flash,previous_lat,long_flash,previous_long)

            if distance_cm>20:   ##if ditance higher than 20cm
                if start_point_inner+int(distance_cm/20)<200:   ###we only have 200 points, should not be higher
                    start_point_inner+=int(distance_cm/20)     ###jumping step to go to next step
                previous_long = long_flash
                previous_lat = lat_flash

            WI_report = coordinates_beam_powers[list(coordinates_beam_powers.keys())[start_point_inner]]
            # print("WI_report",WI_report)
            s,p = wi_arrange(WI_report)
            index_dec_wi = [i[0] for i in sorted(enumerate(p), key=lambda k: k[1], reverse=True)]   ##negative values
            s_sorted = [s[sorting] for sorting in index_dec_wi]
            p_sorted = [p[sorting] for sorting in index_dec_wi]
            for k in range(1,10):
                if any(x in within_thershold_range for x in s_sorted[:k]):
                    found = True
                    break
            if found==False:
                failuar+=1
            counter+=1
        startpoint_probaility[str(start_point)]=1-(failuar/counter)
    probabilities[flash_csv] = startpoint_probaility
print("percentage outlier",ouliers/total_samples)

prob_per_csv = []
for p in probabilities.keys():
    print("per cvs",p,max(list(probabilities[p].values())))
    prob_per_csv.append(max(list(probabilities[p].values())))
print("prob_per_csv",prob_per_csv)
print("average over all",sum(prob_per_csv)/len(prob_per_csv))




######################best correlation is found to maximize the top-10 accuracy, optimization
# print("probabilities",probabilities)
print("***************Optimization step")
position_k_success = {}
for p in range(200):
    inner_dict = {}
    for inc in range(1,35):
        inner_dict[inc] = 0
    position_k_success[p] = inner_dict


for flash_csv in all_csv_flash:
    data_plus_wi = []
    # print("probabilities[flash_csv]",flash_csv,probabilities[flash_csv])
    best_correlation_step = int(max(probabilities[flash_csv], key=probabilities[flash_csv].get))
    print("best_correlation_step",best_correlation_step)
    start_point_inner = best_correlation_step


    ############added
    print("**************************************************",flash_csv)
    with open(flash_csv, 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f)
        data_flash = list(reader)
    ###outliear
    if ouliear_remove is True:
        data_flash_without_outliear = []
        for d in range(1,len(data_flash)):
            if distance(float(data_flash[d][8]),tx_lat,float(data_flash[d][9]),tx_long)<distance_limit_tx:   ###remove outliers
                data_flash_without_outliear.append(data_flash[d])
            else:
                ouliers+=1
        data_flash_without_outliear.insert(0,data_flash[0])  ##adding the header to keep the structure
    else:
        data_flash_without_outliear = data_flash

    ###arranging
    if arranging is True:
        distance_from_tx_data = {}
        origin_lat, origin_long = float(data_flash_without_outliear[1][8]) , float(data_flash_without_outliear[1][9])
        for d in range(1,len(data_flash_without_outliear)):
            lat_flash = float(data_flash_without_outliear[d][8])
            long_flash = float(data_flash_without_outliear[d][9])
            distance_from_tx_data[d] = distance(lat_flash,origin_lat,long_flash,origin_long)
        all_distances = list(distance_from_tx_data.values())
        indexes_decreasing = sorted(range(len(all_distances)), key=lambda k: all_distances[k],reverse=False)
        data_flash_arranged = []
        for i in indexes_decreasing:
            data_flash_arranged.append(data_flash_without_outliear[i+1])
        data_flash_arranged.insert(0,data_flash_without_outliear[0])  ##adding the header to keep the structure
    else:
        data_flash_arranged = data_flash_without_outliear

    print("compare lens",len(data_flash),len(data_flash_without_outliear),len(data_flash_arranged))
    ############added

    for d in data_flash_arranged[1:]:
        lat_flash = float(d[8])
        long_flash = float(d[9])
        rssi_flash = ast.literal_eval(d[15])
        snr_flash = ast.literal_eval(d[19])
        ids_flash = ast.literal_eval(d[16])

        #####remove dplicates of rssi/power measurments
        counting = Counter(ids_flash)
        unique_ids = list(counting.keys())
        unique_snr = []
        unique_rssi = []
        for u in unique_ids:
            repeats = [i for i in range(len(ids_flash)) if ids_flash[i]==u]
            ###only keep maximum
            unique_snr.append(max([snr_flash[r] for r in repeats]))
            unique_rssi.append(max([rssi_flash[r] for r in repeats]))

        ###determin the flash sectors within thershold
        within_thershold_range = []
        for f in range(len(unique_snr)):
            if unique_snr[f]>=unique_snr[unique_rssi.index(max(unique_rssi))]-thershold:    ###snr of the best sector in flash minues thershold
                within_thershold_range.append(unique_ids[f])

        #####mapping
        if previous_long==0:
            previous_long = long_flash
            previous_lat = lat_flash
            distance_cm = 0
        else:
            distance_cm = 100000*distance(lat_flash,previous_lat,long_flash,previous_long)

        if distance_cm>20:   ##if ditance higher than 20cm
            if start_point_inner+int(distance_cm/20)<200:   ###we only have 200 points, should not be higher
                start_point_inner+=int(distance_cm/20)     ###jumping step to go to next step
            previous_long = long_flash
            previous_lat = lat_flash
        # print("start_point_inner",start_point_inner)
        WI_report = coordinates_beam_powers[list(coordinates_beam_powers.keys())[start_point_inner]]
        # print("WI_report",WI_report)
        s,p = wi_arrange(WI_report)

        data_plus_wi.append(d+[s]+[p])
        header = data_flash_arranged[0]+['sector_WI']+['powers_WI']

    print("d,data_plus_wi",len(data_flash),len(data_plus_wi))
    with open(flash_csv.split('.')[0] +'_'+twin+'_WI.csv', 'w', encoding='UTF8', newline='') as f:    ### last two alphabets are GT in all directories
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_plus_wi)
