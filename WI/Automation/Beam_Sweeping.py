import csv
from time import sleep
from tkinter.ttk import Treeview
from numpy import double
import pywinauto
from pywinauto.application import Application
from pywinauto.keyboard import send_keys

# Start Wireless Insite Simulation software
app = Application().start('C:/Program Files/Remcom/Wireless InSite 3.3.5/bin/InSite', timeout=10) 

app.windows()

dlg = app['Wireless InSite 3.3.5.6 - Main:']

#Open a particular project
project_menu = dlg.menu_select("Project -> Open -> Project")

open_dlg = app.window(title_re=".*Open*")
open_dlg.FileNameEdit.set_edit_text("Cat_1")
open_dlg.Open.click()

sleep(10)
header = ['Antenna', 'Max Propogated Power(dBm)']

project_dlg = app.window(title_re=".*Cat_1*")
columns = ['A', 'B', 'C', 'D', 'E', 'F']

#Range over scenarios and receivers to analyze received powers
for column in columns:
    for receiver in range(37,202):
        #with open('Results/Project_c_results/R_L_' + column + '_c_' + str(receiver) +'.csv', 'w', encoding='UTF8', newline='') as f:
        with open('Results/just_for_tests/R_X' + str(receiver) +'_test_purposes.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            project_dlg.TabControlOutput.select('Transmitters / Receivers')
            #item = project_dlg.ListView.get_item('R_L_' + column + '_c_' + str(receiver))

            #Activate receivers sequentially along the trajectory
            item = project_dlg.ListView.get_item('R_X' + str(receiver))
            item.click_input('right')

            dsk = pywinauto.Desktop(backend='win32')
            dsk.PopupMenu.wait('ready').menu().get_menu_path('Active')[0].click_input()
            #dsk.Context.Active.click_input()

            #Sweep over the 34 transmitter antenna patterns
            for antenna in range(12, 36):
                if (antenna == 34):
                    continue

                project_dlg.TabControlOutput.select('Transmitters / Receivers')
                row = []

                item = project_dlg.ListView.get_item('T_x')
                item.click_input(double = True)

                property_dlg = app.window(title_re=".*Transmitter/Receiver*")
                property_dlg.Button2.click()

                tx_property_dlg = app.window(title_re=".*Transmitter properties*")
                antenna = tx_property_dlg.ComboBox1.select(antenna).selected_text()
                row.append(antenna)
                tx_property_dlg.OK.click()

                property_dlg.Apply.click()
                property_dlg.OK.click()
                
                #Run the simulation
                project_dlg.menu_select("Project -> Run -> New")
                sleep(2)
                send_keys("{ENTER}")

                sleep(70)

                #Observe power at active receiver and store it in a csv file
                project_dlg.TabControlOutput.select('Output')

                output_win = project_dlg.child_window(title='Output')

                ctrl = output_win 
                ctrl.Treeview.get_item([u'Area: StudyArea_1']).select()  #Click on a top level element
                ctrl.Treeview.get_item([u'Area: StudyArea_1', u'Point to multipoint']).select()
                ctrl.Treeview.get_item([u'Area: StudyArea_1', u'Point to multipoint', u'Propagation paths']).select()
                ctrl.Treeview.get_item([u'Area: StudyArea_1', u'Point to multipoint', u'Propagation paths', u'T_x']).select()
                #Nov16#ctrl.Treeview.get_item([u'Area: StudyArea_1', u'Point to multipoint', u'Propagation paths', u'T_x_Nov16']).select()
                #rx_ctrl = ctrl.Treeview.get_item([u'Area: StudyArea_1', u'Point to multipoint', u'Propagation paths', u'T_x', u'R_L_' + column + '_c_' + str(receiver)]).select()
                rx_ctrl = ctrl.Treeview.get_item([u'Area: StudyArea_1', u'Point to multipoint', u'Propagation paths', u'T_x', u'R_X' + str(receiver)]).select()
                #Nov16#rx_ctrl = ctrl.Treeview.get_item([u'Area: StudyArea_1', u'Point to multipoint', u'Propagation paths', u'T_x_Nov16', u'R_X' + str(receiver)]).select()
                rx_ctrl.ensure_visible().click_input('right')

                dsk.PopupMenu.wait('ready').menu().get_menu_path('Properties')[0].click_input()

                sleep(3)

                result_dlg = app.window(title_re = ".*Output file properties*")
                row.append(result_dlg.Edit3.window_text())
                result_dlg.OK.click()

                writer.writerow(row)

            #Deactivate the selected receiver to proceed further
            project_dlg.TabControlOutput.select('Transmitters / Receivers')
            #item = project_dlg.ListView.get_item('R_L_' + column + '_c_' + str(receiver))
            item = project_dlg.ListView.get_item('R_X' + str(receiver))
            item.click_input('right')

            dsk = pywinauto.Desktop(backend='win32')
            dsk.PopupMenu.wait('ready').menu().get_menu_path('Active')[0].click_input()




