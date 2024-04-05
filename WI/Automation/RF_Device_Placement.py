import csv
from time import sleep
from tkinter.ttk import Treeview
from unicodedata import name
from numpy import double
import pywinauto
import csv
from pywinauto import mouse
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import pandas as pd

# Start Wireless Insite Simulation software
app = Application().start('C:/Program Files/Remcom/Wireless InSite 3.3.5/bin/InSite', timeout=10) 

app.windows()

dlg = app['Wireless InSite 3.3.5.6 - Main:']

#Open a particular project
project_menu = dlg.menu_select("Project -> Open -> Project")

open_dlg = app.window(title_re=".*Open*")
open_dlg.FileNameEdit.set_edit_text("Cat_1")
open_dlg.Open.click()

# Open file containing the height of receivers
sleep(10)
project_dlg = app.window(title_re=".*Wireless*")
my_counter = 0
altitude_file = 'opposite_lane_200_points_withHeights.csv'
alt_csv = pd.read_csv(altitude_file)
altitudes = alt_csv['Set Altitude']

# Open the device co-ordinates file
with open('opposite_lane_200_points.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
            project_dlg.TabControlOutput.select('Transmitters / Receivers')
            project_dlg.click_input(button='right')

            #Classify as Transmitter or Receiver
            if row[1][0] == 'T':
                app2 = pywinauto.Desktop(backend='win32')
                app2.PopupMenu.wait('ready').menu().get_menu_path('New -> Transmitter Set -> Points')[2].click_input()
                #project_dlg.menu_select("Project -> New -> Transmitter Set -> Points")
            elif row[1][0] == 'R':
                project_dlg.menu_select("Project -> New -> Receiver Set -> Points")

            # Place the device using the appropriate co-ordinates
            map_win = app.window(title_re=".*Project view:*")
            map_win['StandardWindow0'].click_input(button = 'left')
            map_win['StandardWindow0'].click_input(button = 'right')

            parameter_win = app.window(title_re=".*Transmitter/Receiver properties*")

            parameter_win['Edit'].set_text(row[1])
            parameter_win['Origin longitude: StandardWindow'].Edit3.set_text(row[3])
            parameter_win['Origin latitude: StandardWindow'].Edit3.set_text(row[2])

            # Assign the receiver a standard antenna
            if row[1][0] == 'T':
                parameter_win.Button2.click()
                property_dlg = app.window(title_re=".*Transmitter properties*")
            elif row[1][0] == 'R':
                parameter_win.Button3.click()
                property_dlg = app.window(title ="Receiver properties")
                property_dlg.ComboBox.select("Legacy_Rx")

            ######property_dlg.print_control_identifiers()
            # Set waveform and orientations
            property_dlg.ComboBox2.select("Sinusoid_60Ghz_ad")
            if row[1][0] == 'T':
                property_dlg.Edit0.set_text(50)
            elif row[1][0] == 'R':
                property_dlg.Edit0.set_text(230)
                #property_dlg.Edit2.set_text(110)
            property_dlg.OK.click()

            # Set the altitude for each device
            parameter_win.Button4.click()
            layout_dlg = app.window(title ="Layout properties")
            layout_dlg.Button0.click()
            dsk = pywinauto.Desktop(backend='win32')
            item = dsk['View/ edit vertices']['Double-click to editListView']
            item.click()
            send_keys('{UP}')
            send_keys('{VK_SHIFT down}{F10}{VK_SHIFT up}')
            dsk.PopupMenu.wait('ready').menu().get_menu_path('Delete vertex')[0].click_input()

            item.click('right')
            dsk.PopupMenu.wait('ready').menu().get_menu_path('New vertex')[0].click_input()

            add_vtx_dlg = app.window(title ="Add new vertex")
            add_vtx_dlg['Edit3'].set_text(altitudes[my_counter])
            add_vtx_dlg.Button2.click()

            item.click()
            send_keys('{ENTER}')
            layout_dlg.OK.click()
            
            my_counter += 1
            parameter_win.Apply.click()
            parameter_win.OK.click()