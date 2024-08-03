import random
import csv
import select_com_bn_nn as sc
#產生虛擬測試資料來測試神經網路以及基因演算法

data = sc.training_design_parameter(20)

#選擇設計參數組合/產生訓練資料

circuit_data_sw = []
circuit_data_indl = []
circuit_data_cl = []
circuit_data_ripple = []

for param in data:
    cap_val = param['cap']
    ind_val = param['ind']
    fsw_val = param['fsw']

    #假關係式 (測試1:以Random(gamma distribution)方式產生 預想1:神經網路會學習到這種隨機模式)
    shape_par = 9
    scale_par = 0.5
    #switch loss [high-side low-side]
    high_side_loss = random.gammavariate(shape_par, scale_par)
    low_side_loss = random.gammavariate(shape_par, scale_par)/10
    #inductor loss [loss_cu loss_fe] *暫不考慮磁心損耗
    ind_loss_cu = random.gammavariate(shape_par, scale_par)
    ind_loss_fe = random.gammavariate(shape_par, scale_par)
    #capacitor loss [cap loss]
    cap_loss = random.gammavariate(shape_par, scale_par)
    #ripples [voltage current]
    ripple_v = random.gammavariate(shape_par, scale_par)
    ripple_c = random.gammavariate(shape_par, scale_par)
    circuit_data_sw.append([fsw_val,cap_val,ind_val,high_side_loss,low_side_loss])
    circuit_data_cl.append([fsw_val,cap_val,ind_val,cap_loss])
    circuit_data_indl.append([fsw_val,cap_val,ind_val,ind_loss_cu])
    circuit_data_ripple.append([fsw_val,cap_val,ind_val,ripple_v,ripple_c])

#訓練資料存至csv檔,供後續神經網路訓練
#1:switch losses training data
with open('sw_pseudo.csv', 'w', newline='') as csvfile_sw:
    fieldnames = ['fsw', 'cap', 'ind', 'pl_sw_h', 'pl_sw_l']
    writer = csv.DictWriter(csvfile_sw, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(circuit_data_sw)):
        writer.writerow({'fsw': circuit_data_sw[i][0], 'cap': circuit_data_sw[i][1], 'ind': circuit_data_sw[i][2], 'pl_sw_h': circuit_data_sw[i][3], 'pl_sw_l': circuit_data_sw[i][4]})

#2:inductor loss training data
with open('indl_pseudo.csv', 'w', newline='') as csvfile_indl:
    fieldnames = ['fsw', 'cap', 'ind', 'pl_i_cu']
    writer = csv.DictWriter(csvfile_indl, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(circuit_data_indl)):
        writer.writerow({'fsw': circuit_data_indl[i][0], 'cap': circuit_data_indl[i][1], 'ind': circuit_data_indl[i][2], 'pl_i_cu': circuit_data_indl[i][3]})
#3:capacitor loss training data
with open('cl_pseudo.csv', 'w', newline='') as csvfile_cl:
    fieldnames = ['fsw', 'cap', 'ind', 'pl_c']
    writer = csv.DictWriter(csvfile_cl, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(circuit_data_cl)):
        writer.writerow({'fsw': circuit_data_cl[i][0], 'cap': circuit_data_cl[i][1], 'ind': circuit_data_cl[i][2], 'pl_c': circuit_data_cl[i][3]})
#4:ripples training data
with open('ripple_pseudo.csv', 'w', newline='') as csvfile_rp:
    fieldnames = ['fsw', 'cap', 'ind', 'ripple_v', 'ripple_c']
    writer = csv.DictWriter(csvfile_rp, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(circuit_data_ripple)):
        writer.writerow({'fsw': circuit_data_ripple[i][0], 'cap': circuit_data_ripple[i][1], 'ind': circuit_data_ripple[i][2], 'ripple_v': circuit_data_ripple[i][3], 'ripple_c': circuit_data_ripple[i][4]})