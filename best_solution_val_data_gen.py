import func_lib as fl

def best_solution_val_data_gen(fsw,cap,ind):
    # Fig. 18.(a) 
    result = []
    # 計算ind esr
    l_range = {'l_min':100e-6,'l_max':1000e-6}
    # disfactor
    dis_factor = 0.14
    # ksel
    ksel = 4.389*1e-11
    # fsw 左右共幾個測試點
    point = 8
    # 每點間格距離
    space = 0.5
    # 計算範圍
    ran = point*space/2/10
    # 最小範圍
    min_ran = 1 - ran
    # 頻率起始倍率
    start_rate = min_ran
    new_rate = start_rate
    fsw_var_range = [0.8*fsw, 0.85*fsw, 0.9*fsw, 0.95*fsw, fsw, 1.05*fsw, 1.1*fsw, 1.15*fsw, 1.2*fsw]
    for fsw_temp in fsw_var_range:
        fsw_temp = round(fsw_temp, -2)
        print(fsw_temp)
        #計算電容esr
        cap_esr = fl.cap_esr_cal(fsw_temp,cap,dis_factor)
        #計算電容esl
        cap_esl = fl.cap_esl_cal(cap,ksel)
        #計算電感esr
        ind_esr = fl.ind_esr_cal(ind,l_range)

        design_param = {'fsw':0.0,'cap':0.0,'cap_esr':0.0,'cap_esl':0.0,'ind':0.0,'ind_esr':0.0}
        design_param['cap'] = cap
        design_param['cap_esl'] = cap_esl
        design_param['cap_esr'] = cap_esr
        design_param['fsw'] = fsw_temp
        design_param['ind'] = ind
        design_param['ind_esr'] = ind_esr
        result.append(design_param)
        new_rate = new_rate + space
    #Fig. 18. (b) inductor var
    #變化範圍
    ind_var_range = [134.1e-6,156.33e-6,188.3e-6, ind, 232.5e-6, 251.47e-6, 281.32e-6]
    for ind_temp in ind_var_range:
        ind_temp = round(ind_temp, 7)
        #只有inductor esr會變化
        print(ind_temp)
        #計算電容esr
        cap_esr = fl.cap_esr_cal(fsw,cap,dis_factor)
        #計算電容esl
        cap_esl = fl.cap_esl_cal(cap,ksel)
        #計算電感esr
        ind_esr = fl.ind_esr_cal(ind_temp,l_range)

        design_param = {'fsw':0.0,'cap':0.0,'cap_esr':0.0,'cap_esl':0.0,'ind':0.0,'ind_esr':0.0}
        design_param['cap'] = cap
        design_param['cap_esl'] = cap_esl
        design_param['cap_esr'] = cap_esr
        design_param['fsw'] = fsw
        design_param['ind'] = ind_temp
        design_param['ind_esr'] = ind_esr
        result.append(design_param)
    # (c) capacitor變化
    #變化範圍
    cap_var_range = [210e-6, 248.1e-6, cap, 282e-6, 320e-6]
    for cap_temp in cap_var_range:
        #只有inductor esr會變化
        cap_temp = round(cap_temp, 7)
        print(cap_temp)
        #計算電容esr
        cap_esr = fl.cap_esr_cal(fsw,cap_temp,dis_factor)
        #計算電容esl
        cap_esl = fl.cap_esl_cal(cap_temp,ksel)
        #計算電感esr
        ind_esr = fl.ind_esr_cal(ind,l_range)

        design_param = {'fsw':0.0,'cap':0.0,'cap_esr':0.0,'cap_esl':0.0,'ind':0.0,'ind_esr':0.0}
        design_param['cap'] = cap_temp
        design_param['cap_esl'] = cap_esl
        design_param['cap_esr'] = cap_esr
        design_param['fsw'] = fsw
        design_param['ind'] = ind
        design_param['ind_esr'] = ind_esr
        result.append(design_param)

    return result


            

    