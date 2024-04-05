########################################################
##### Project: digital twin FLASH
##### Contact: Batool Salehi
##### optimization problem in the paper
##### Change inputs line 25-34 for each setting (LOS/NLOS). The inputs are the pickle files from optimize.py
########################################################
import pickle
import math
from random import randrange

from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import Counter
from scipy import stats
from sklearn.metrics import ndcg_score
import random
import math
from collections import Counter
import numpy as np
import time

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (6,4)

###inputs
with open('pkls/Twin1_LOS.pkl', 'rb') as f:
    Twin1 = pickle.load(f)

with open('pkls/Twin2_LOS.pkl', 'rb') as f:
    Twin2 = pickle.load(f)

with open('pkls/Twin3_LOS.pkl', 'rb') as f:
    Twin3 = pickle.load(f)
maximum_K = 34+1    #11 for new table, be carefull to plus 1

record_time = []
probs_alpha = []
bm_time_alpha = []
usage_alpha = []
objective_alpha = []
for alpha in np.arange(0, 0.1, 0.1):  #was 20
    collect_probablities = []
    collect_bm_time = []
    collect_objectives = []
    usage_1 = 0
    usage_2 = 0
    usage_3 = 0

    for l in range(200): ##200 locations
        print("#############################")
        objective_for_twins = []
        best_k = {}
        for i in range(3): ##three twins
            if i==0:
                twin = Twin1
                t_comp = 0 #(4*34)   #x
            elif i==1:
                twin = Twin2
                t_comp = 0 #(2*34*200) #100x
            elif i==2:
                twin = Twin3
                t_comp = 0 #(4*34*200)  #200x

            prob = twin[l]  ###a dictionary for K sectors twin T and location l

            objective_per_k = []
            begin = time.time()
            for k in range(1,maximum_K):
                t_comm_max = 20*math.floor((34-1)/32)+0.156*(1+(34-1)%32)
                t_comm = 20*math.floor((k-1)/32)+0.156*(1+(k-1)%32)
                t_comp_max = (4*34*200)
                # print("alpha",alpha,"k:",k,"prob[k]",prob[k],"t_comm",t_comm,"t_comp",t_comp,"t_comm/t_comm_max",t_comm/t_comm_max,"t_comp/t_comp_max",t_comp/t_comp_max)
                if prob[k]!=0:
                    # objective_per_k.append(prob[k]+alpha*((t_comm_max-t_comm)/t_comm_max)+(1 - alpha)*((t_comp_max-t_comp)/t_comp_max))#+t_comp/t_comp_max)))
                    # objective_per_k.append(prob[k]+alpha*((t_comp*t_comm)/((4*34*200)*t_comm_max)))
                    # objective_per_k.append(prob[k]+alpha*(((t_comm_max-t_comm)/t_comm_max)*((t_comp_max-t_comp)/t_comp_max)))
                    objective_per_k.append(prob[k]+alpha*(((t_comm_max-t_comm)/t_comm_max)*((t_comp_max-t_comp)/t_comp_max)))
                    # objective_per_k.append(prob[k]+alpha/2*(((t_comm_max-t_comm)/t_comm_max)+((t_comp_max-t_comp)/t_comp_max)))
                else:
                    objective_per_k.append(0)
            end = time.time()
            record_time.append(end-begin)
            ###find the best K for objective
            best_k[i] = objective_per_k.index(max(objective_per_k))+1
            objective_for_twins.append(max(objective_per_k))
        print("check per twin",best_k,objective_for_twins)
        ###################################
        selected_maxes = [index for index, item in enumerate(objective_for_twins) if item == max(objective_for_twins)]
        if selected_maxes!=[0,0,0]:
            print("selected_maxes",selected_maxes)
            if len(selected_maxes)>1:
                selected_twin = selected_maxes[-1]#[randrange(len(selected_maxes))]
            else:
                selected_twin = selected_maxes[0]   ###list

            ####twin usage
            if selected_twin ==0:
                collect_probablities.append(Twin1[l][best_k[selected_twin]])
                usage_1+=1
            elif selected_twin ==1:
                collect_probablities.append(Twin2[l][best_k[selected_twin]])
                usage_2+=1
            elif selected_twin ==2:
                collect_probablities.append(Twin3[l][best_k[selected_twin]])
                usage_3+=1
            collect_objectives.append(objective_for_twins[selected_twin])
            collect_bm_time.append(20*math.floor((best_k[selected_twin]-1)/32)+0.156*(1+(best_k[selected_twin]-1)%32))
    print("collect_probablities",collect_probablities,sum(collect_probablities)/len(collect_probablities))
    usage_alpha.append([usage_1/200,usage_2/200,usage_3/200])
    probs_alpha.append(sum(collect_probablities)/len(collect_probablities))
    bm_time_alpha.append(sum(collect_bm_time)/len(collect_bm_time))
    objective_alpha.append(sum(collect_objectives)/len(collect_objectives))


print("t_comm_max",20*math.floor((maximum_K-1)/32)+0.156*(1+(maximum_K-1)%32))
print("usage_alpha",usage_alpha)
print("objective_alpha",objective_alpha,objective_alpha.index(max(objective_alpha)))
print("probs_alpha",probs_alpha)
print("bm_time_alpha",bm_time_alpha)
# p
print("avg record_time",sum(record_time)/len(record_time))
# print("Twin3",Twin3)
# print("Twin1",sum([Twin1[k][10] for k in Twin1.keys()])/len([Twin1[k][10] for k in Twin1.keys()]))
# print("Twin2",sum([Twin2[k][10] for k in Twin2.keys()])/len([Twin2[k][10] for k in Twin2.keys()]))
# print("Twin3",sum([Twin3[k][10] for k in Twin3.keys()])/len([Twin3[k][10] for k in Twin3.keys()]))
# print(Twin1[40])
# print(Twin2[40])
# print(Twin3[40])



# avg_over_l = {}
# for k in range(1,35):
#     avg = 0
#     for l in Twin3.keys():
#         avg+=Twin3[l][k]
#     avg_over_l[k] = avg/200
# print(avg_over_l)

