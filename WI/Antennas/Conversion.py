"""
Step - 3
Code for generating antenna pattern files for WI
* This code is for Legacy Measurements
"""


import pandas as pd
import math
import scipy.io

legacy_data_location_3D = '/Users/utkudemir/Desktop/Research/Wireless/Wireless_Insite/talon-sector-patterns-master/legacy_measurements/sectorpattern_3d_sta.mat';
p_sta_3D = scipy.io.loadmat(legacy_data_location_3D)
legacy_data_location_3D_ap = '/Users/utkudemir/Desktop/Research/Wireless/Wireless_Insite/talon-sector-patterns-master/legacy_measurements/sectorpattern_3d_ap.mat';
p_sta_3D_ap = scipy.io.loadmat(legacy_data_location_3D_ap)

#####    #####    #####    #####  TRANSMITTERS  #####    #####    #####    #####    #####

'''
for which_antenna in range(34):
    print('antenna - ', which_antenna)

    df = pd.DataFrame(columns = ['theta', 'phi', 'gain_theta', 'gain_phi'])

    theta_list = []
    phi_list = []
    gainTheta_list = []
    gainPhi_list = []
    tracker_phi = 0
    tracker_theta = 0

    for theta in range(90,50,-4):
        theta = float(theta)

        for phi in range(-100,102,2):

            phi = float(phi)
            gainPhi_list.append("{:.6f}".format(0.0))

            if theta == 0:
                theta_list.append("{:.6f}".format(theta))
            elif int(math.log10(theta)) + 1 == 1:
                theta_list.append("{:.6f}".format(theta))
            elif int(math.log10(theta)) + 1 == 2:
                theta_list.append("{:.5f}".format(theta))
            elif int(math.log10(theta)) + 1 == 3:
                theta_list.append("{:.4f}".format(theta))

            if phi == 0.0:
                phi_list.append("{:.6f}".format(phi))
            elif phi > 0.0:
                if int(math.log10(phi)) + 1 == 1:
                    phi_list.append("{:.6f}".format(phi))
                elif int(math.log10(phi)) + 1 == 2:
                    phi_list.append("{:.5f}".format(phi))
                elif int(math.log10(phi)) + 1 == 3:
                    phi_list.append("{:.4f}".format(phi))
            else:
                if int(math.log10(-phi)) + 1 == 1:
                    phi_list.append("{:.5f}".format(phi))
                elif int(math.log10(-phi)) + 1 == 2:
                    phi_list.append("{:.4f}".format(phi))
                elif int(math.log10(-phi)) + 1 == 3:
                    phi_list.append("{:.3f}".format(phi))

            add_number = p_sta_3D['pattern_snr'][:, :, which_antenna][tracker_theta, tracker_phi]

            if math.isnan(add_number):
                add_number = 0.0
                gainTheta_list.append("{:.6f}".format(add_number))
            elif add_number == 0.0:
                gainTheta_list.append("{:.6f}".format(add_number))
            elif add_number < 0.0:
                if int(math.log10(-add_number)) <= 0:
                    gainTheta_list.append("{:.5f}".format(add_number))
                elif int(math.log10(-add_number)) + 1 == 2:
                    gainTheta_list.append("{:.4f}".format(add_number))
                elif int(math.log10(-add_number)) + 1 == 3:
                    gainTheta_list.append("{:.3f}".format(add_number))
                else:
                    print('wtf ', add_number)
            else:
                if int(math.log10(add_number)) == 0 or int(math.log10(add_number)) + 1 == 0:
                    gainTheta_list.append("{:.6f}".format(add_number))
                elif int(math.log10(add_number)) + 1 == 2 or int(math.log10(add_number)) + 1 == -1:
                    gainTheta_list.append("{:.5f}".format(add_number))
                elif int(math.log10(add_number)) + 1 == 3 or int(math.log10(add_number)) + 1 == -2:
                    gainTheta_list.append("{:.4f}".format(add_number))
                elif int(math.log10(add_number)) + 1 < -2:
                    add_number = 0.0
                    gainTheta_list.append("{:.6f}".format(add_number))
                else:
                    print('wtf ', add_number)
            tracker_phi += 1

        tracker_phi = 0
        tracker_theta += 1

    df['theta'] = theta_list
    df['phi'] = phi_list
    df['gain_theta'] = gainTheta_list
    df['gain_phi'] = gainPhi_list

    #print(df)
    if which_antenna < 11:

        df.to_csv('/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/AntPat_legacy0' + str(which_antenna+1) + '_direct_snr.txt', header=None, index=None, sep=' ')

        with open('/Users/utkudemir/Desktop/Talon_AntennaPatterns/AntennaTemplate_legacy.txt') as fp:
            AntTemp = fp.read()
        with open(
                '/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/AntPat_legacy0' + str(which_antenna+1) + '_direct_snr.txt') as fp:
            AntPat = fp.read()
        AntTemp += "\n"
        AntTemp += AntPat

        with open(
                '/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/rotated_AntPat_legacy0' + str(which_antenna+1) + '_direct_snr.uan', 'w') as fp:
            fp.write(AntTemp)

    else:
        df.to_csv('/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/AntPat_legacy' + str(which_antenna+1) + '_direct_snr.txt', header=None, index=None, sep=' ')

        with open('/Users/utkudemir/Desktop/Talon_AntennaPatterns/AntennaTemplate_legacy.txt') as fp:
            AntTemp = fp.read()
        with open(
                '/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/AntPat_legacy' + str(which_antenna+1) + '_direct_snr.txt') as fp:
            AntPat = fp.read()
        AntTemp += "\n"
        AntTemp += AntPat

        with open(
                '/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/rotated_AntPat_legacy' + str(which_antenna+1) + '_direct_snr.uan', 'w') as fp:
            fp.write(AntTemp)

'''

#####    #####    #####    #####  RECEIVER  #####    #####    #####    #####    #####
Rx_antenna = 33
df = pd.DataFrame(columns = ['theta', 'phi', 'gain_theta', 'gain_phi'])

theta_list = []
phi_list = []
gainTheta_list = []
gainPhi_list = []
tracker_phi = 0
tracker_theta = 0

for theta in range(90,50,-4):
    theta = float(theta)

    for phi in range(-100,102,2):

        phi = float(phi)
        gainPhi_list.append("{:.6f}".format(0.0))

        if theta == 0:
            theta_list.append("{:.6f}".format(theta))
        elif int(math.log10(theta)) + 1 == 1:
            theta_list.append("{:.6f}".format(theta))
        elif int(math.log10(theta)) + 1 == 2:
            theta_list.append("{:.5f}".format(theta))
        elif int(math.log10(theta)) + 1 == 3:
            theta_list.append("{:.4f}".format(theta))

        if phi == 0.0:
            phi_list.append("{:.6f}".format(phi))
        elif phi > 0.0:
            if int(math.log10(phi)) + 1 == 1:
                phi_list.append("{:.6f}".format(phi))
            elif int(math.log10(phi)) + 1 == 2:
                phi_list.append("{:.5f}".format(phi))
            elif int(math.log10(phi)) + 1 == 3:
                phi_list.append("{:.4f}".format(phi))
        else:
            if int(math.log10(-phi)) + 1 == 1:
                phi_list.append("{:.5f}".format(phi))
            elif int(math.log10(-phi)) + 1 == 2:
                phi_list.append("{:.4f}".format(phi))
            elif int(math.log10(-phi)) + 1 == 3:
                phi_list.append("{:.3f}".format(phi))

        add_number = p_sta_3D_ap['pattern_snr'][:, :, Rx_antenna][tracker_theta, tracker_phi]

        if math.isnan(add_number):
            add_number = 0.0
            gainTheta_list.append("{:.6f}".format(add_number))
        elif add_number == 0.0:
            gainTheta_list.append("{:.6f}".format(add_number))
        elif add_number < 0.0:
            if int(math.log10(-add_number)) <= 0:
                gainTheta_list.append("{:.5f}".format(add_number))
            elif int(math.log10(-add_number)) + 1 == 2:
                gainTheta_list.append("{:.4f}".format(add_number))
            elif int(math.log10(-add_number)) + 1 == 3:
                gainTheta_list.append("{:.3f}".format(add_number))
            else:
                print('wtf ', add_number)
        else:
            if int(math.log10(add_number)) == 0 or int(math.log10(add_number)) + 1 == 0:
                gainTheta_list.append("{:.6f}".format(add_number))
            elif int(math.log10(add_number)) + 1 == 2 or int(math.log10(add_number)) + 1 == -1:
                gainTheta_list.append("{:.5f}".format(add_number))
            elif int(math.log10(add_number)) + 1 == 3 or int(math.log10(add_number)) + 1 == -2:
                gainTheta_list.append("{:.4f}".format(add_number))
            elif int(math.log10(add_number)) + 1 < -2:
                add_number = 0.0
                gainTheta_list.append("{:.6f}".format(add_number))
            else:
                print('wtf ', add_number)
        tracker_phi += 1

    tracker_phi = 0
    tracker_theta += 1

df['theta'] = theta_list
df['phi'] = phi_list
df['gain_theta'] = gainTheta_list
df['gain_phi'] = gainPhi_list

#print(df)

df.to_csv('/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/AntPat_legacyRx_direct_snr.txt', header=None, index=None, sep=' ')

with open('/Users/utkudemir/Desktop/Talon_AntennaPatterns/AntennaTemplate_legacy.txt') as fp:
    AntTemp = fp.read()

with open('/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/AntPat_legacyRx_direct_snr.txt') as fp:
    AntPat = fp.read()
    AntTemp += "\n"
    AntTemp += AntPat

with open('/Users/utkudemir/Desktop/Talon_AntennaPatterns/Talon_AntennaPatterns_rotated_legacy/rotated_AntPat_legacyRx_direct_snr.uan', 'w') as fp:
    fp.write(AntTemp)

