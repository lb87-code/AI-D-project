from PyLTSpice import SimRunner
from PyLTSpice import SpiceEditor
from PyLTSpice.log.ltsteps import LTSpiceLogReader
from PyLTSpice.sim.process_callback import ProcessCallback  # Importing the ProcessCallback class type
import select_com_bn_nn as sc
import func_lib as fl
import csv

# Force another simulatior
simulator = r"C:\Program Files\ADI\LTspice\LTspice.exe"

#設計參數儲存方式[fsw , cap_part_no , capacitance (uF) , volume (cm^3) , ESR (歐姆) , ESL (H) , core_part_no , inductance (uH) , RL (歐姆)] 
#               [0]     [1]           [2]               [3]             [4]         [5]         [6]           [7]               [8]
#取得隨機產生的組合 #20 x 20 x 20 = 8000
datas = sc.training_design_parameter(2)
#儲存設計參數[fsw, capacitance, inductance]值
design_param = []
#儲存元件能量損耗以及電壓跟電流
#losses of switch high-side, low-side
pl_sw_h = []
pl_sw_l = []
#losses of inductor (暫時忽略磁心損耗)
pl_i_cu = []
#capacitor loss
pl_c = []
#Ripples voltage, current
vout_max = []
vout_min = []
il_max = []
il_min = []
vout_avg =[]
il_avg = []
ripple_v = []
ripple_i = []

for data in datas:
    print(data)

# select spice model

LTC = SimRunner(parallel_sims=1, output_folder='./temp')
LTC.create_netlist('./circuitfile/synchronous_buck_va_v3.asc')
netlist = SpiceEditor('./circuitfile/synchronous_buck_va_v3.net')  
# set default arguments
netlist.add_instructions(
    ".meas P_s1 avg V(vin,sourceHS)*Ix(U6:1)",
    ".meas P_s2 avg V(sourcehs)*Ix(U5:1)"
)


for param in datas:
    netlist.set_parameters(fs=param['fsw'])
    netlist.set_parameters(cap1=param['cap'])
    netlist.set_parameters(ind=param['ind'])
    netlist.set_parameters(RL=param['ind_esr'])
    netlist.set_parameters(LC1=param['cap_esl'])
    netlist.set_parameters(RC1=param['cap_esr'])
    design_param.append([param['fsw'], param['cap'], param['ind']])
    print(f"simulating switching frequency: {param['fsw']} Hz capacitance: {param['cap'] } F inductance: {param['ind']} H ")
    LTC.run(netlist)

for raw, log in LTC:
    print("Raw file: %s, Log file: %s" % (raw, log))
    data = LTSpiceLogReader(log)
    meas_names = data.get_measure_names()
    # print(' '.join([f"{name:15s}" for name in meas_names]), end='\n')
    for name in meas_names:
        # print(f"{name}: {data[name]}", end = '\n') #打印所有量測資料
        if name == 'p_s1':
            pl_sw_h.append(data[name])
        elif name == 'p_s2':
            pl_sw_l.append(data[name])
        elif name == 'p_l_cu':
            pl_i_cu.append(data[name])
        elif name == 'p_c':
            pl_c.append(data[name])
        elif name == 'vout_max':
            vout_max.append(data[name])
        elif name == 'vout_min':
            vout_min.append(data[name])
        elif name == 'il_max':
            il_max.append(data[name])
        elif name == 'il_min':
            il_min.append(data[name])
        elif name == 'vout':
            vout_avg.append(data[name])
        elif name == 'il':
            il_avg.append(data[name])
        # else:
        #     print("unknown parameter")

netlist.reset_netlist()

# Sim Statistics
print('Successful/Total Simulations: ' + str(LTC.okSim) + '/' + str(LTC.runno))
# 整合switch loss以及voltage and current ripple
pl_sw = []
ripple =[]
#[high-side , low-side]
pl_sw = fl.com_list(pl_sw_h,pl_sw_l)
#[voltage,current]
ripple_v = fl.find_ripple(vout_max,vout_min,vout_avg)
ripple_i = fl.find_ripple(il_max,il_min,il_avg)
ripple = fl.com_list(ripple_v,ripple_i)
print(f"pl_sw:{pl_sw}")
print(f"{ripple}")
print(f"design_param:{design_param}")
print(f"pl_c:{pl_c}")
print(f"pl_sw_h:{pl_sw_h}")
print(f"pl_sw_l:{pl_sw_l}")
print(f"pl_sw:{pl_sw}")
print(f"pl_i_cu:{pl_i_cu}")
print(f"vout_max:{vout_max}")
print(f"vout_min:{vout_min}")
print(f"vout_avg:{vout_avg}")

#訓練資料存至csv檔 提供後續神經網路訓練
#1:switch losses training data
with open('sw.csv', 'w', newline='') as csvfile_sw:
    fieldnames = ['fsw', 'cap', 'ind', 'pl_sw_h', 'pl_sw_l']
    writer = csv.DictWriter(csvfile_sw, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(design_param)):
        writer.writerow({'fsw': design_param[i][0], 'cap': design_param[i][1], 'ind': design_param[i][2], 'pl_sw_h': pl_sw_h[i][0], 'pl_sw_l': pl_sw_l[i][0]})

#2:inductor loss training data
with open('indl.csv', 'w', newline='') as csvfile_indl:
    fieldnames = ['fsw', 'cap', 'ind', 'pl_i_cu']
    writer = csv.DictWriter(csvfile_indl, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(design_param)):
        writer.writerow({'fsw': design_param[i][0], 'cap': design_param[i][1], 'ind': design_param[i][2], 'pl_i_cu': pl_i_cu[i][0]})
#3:capacitor loss training data
with open('cl.csv', 'w', newline='') as csvfile_cl:
    fieldnames = ['fsw', 'cap', 'ind', 'pl_c']
    writer = csv.DictWriter(csvfile_cl, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(design_param)):
        writer.writerow({'fsw': design_param[i][0], 'cap': design_param[i][1], 'ind': design_param[i][2], 'pl_c': pl_c[i][0]})
#4:ripples training data
with open('ripple.csv', 'w', newline='') as csvfile_rp:
    fieldnames = ['fsw', 'cap', 'ind', 'ripple_v', 'ripple_c']
    writer = csv.DictWriter(csvfile_rp, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(design_param)):
        writer.writerow({'fsw': design_param[i][0], 'cap': design_param[i][1], 'ind': design_param[i][2],'ripple_v': ripple[i][0], 'ripple_c': ripple[i][1]})


enter = input("Press enter to delete created files")
if enter == '':
    LTC.file_cleanup()