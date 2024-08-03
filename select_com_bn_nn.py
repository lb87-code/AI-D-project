import func_lib as fl
import math

def training_design_parameter(num_values):
    #產生設計參數[fs C L]組合給電路模擬軟體

    #fs C L are uniformly selected within [fmin fmax] [Lmin Lmax] [Cmin Cmax] 
    # 8000 combination 20 x 20 x 20

    f_range = {'f_min':20e3,'f_max':200e3}
    c_range = {'c_min':47e-6,'c_max':726e-6}
    l_range = {'l_min':100e-6,'l_max':1000e-6}

    

    result = []

    # ESL factor 4.389*10^-11
    ksel = 4.389*1e-11
    # Dissipation factor
    dis_factor = 0.14
    # assume r(resistance of wire) = 2.14 * 10^-5 歐姆/mm
    r = 2.14e-5

    #計算8000種組合的電容ESR ESL 電感 ESR
    L_max = [509,1003,2519,144]
    L_min = [144,509,1003,30]

    core_data = fl.access_core_data()



    for fsw in fl.cal_step(f_range['f_min'],f_range['f_max'],num_values):
        for cap in fl.cal_step(c_range['c_min'],c_range['c_max'],num_values):
            for ind in fl.cal_step(l_range['l_min'],l_range['l_max'],num_values):
                #計算電容寄生參數
                cap_esr = dis_factor/2.0/math.pi/fsw/cap
                cap_esl = ksel/cap
                #計算電感寄生參數
                if  ind <= 144e-6 and ind >= l_range['l_min']:
                    i = 3
                    N = fl.cal_ind_turns(ind,float(core_data[i][8])*1e-9)
                    OD = float(core_data[i][1])
                    ID = float(core_data[i][2])
                    H = float(core_data[i][3])
                    ind_esr = N*(OD-ID+2*H)*r
                elif ind <= 509e-6 and ind > 144e-6:
                    i = 0
                    N = fl.cal_ind_turns(ind,float(core_data[i][8])*1e-9)
                    OD = float(core_data[i][1])
                    ID = float(core_data[i][2])
                    H = float(core_data[i][3])
                    ind_esr = N*(OD-ID+2*H)*r
                elif ind <= 1003e-6 and ind > 509e-6:
                    i = 1
                    N = fl.cal_ind_turns(ind,float(core_data[i][8])*1e-9)
                    OD = float(core_data[i][1])
                    ID = float(core_data[i][2])
                    H = float(core_data[i][3])
                    ind_esr = N*(OD-ID+2*H)*r
                elif ind <= 2519e-6 and ind > 1003e-6:
                    i = 2
                    N = fl.cal_ind_turns(ind,float(core_data[i][8])*1e-9)
                    OD = float(core_data[i][1])
                    ID = float(core_data[i][2])
                    H = float(core_data[i][3])
                    ind_esr = N*(OD-ID+2*H)*r
                else:
                    print("warnning! inductance out of range")
                design_param = {'fsw':0.0,'cap':0.0,'cap_esr':0.0,'cap_esl':0.0,'ind':0.0,'ind_esr':0.0}
                design_param['cap'] = cap
                design_param['cap_esl'] = cap_esl
                design_param['cap_esr'] = cap_esr
                design_param['fsw'] = fsw
                design_param['ind'] = ind
                design_param['ind_esr'] = ind_esr
                result.append(design_param)
                


    return result
