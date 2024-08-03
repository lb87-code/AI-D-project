import math 
import gen_lookup_table as lookuptable
import random
def design_parameter_return(data_num):
    cap_lookup_table=lookuptable.cap_table_return()

    # for data in cap_lookup_table:
    #     print(data)

    ind_lookup_table=lookuptable.ind_table_return()

    # for data in ind_lookup_table:
    #     print(data)

    #產生開關頻率 20kHz ~ 200kHz
    fsw = []
    fsw_max = 200e3
    fsw_min = 20e3
    fsw_temp= fsw_min
    while True:
        if fsw_temp > fsw_max:
            break
        fsw.append(fsw_temp)
        fsw_temp += 100

    #選出模擬所需要的輸入資料組合 例如 8000筆

    Design_param_combination = []
    for i in range(data_num):
        cap_rand = random.choice(cap_lookup_table)
        ind_rand = random.choice(ind_lookup_table)
        fsw_rand = random.choice(fsw)
        #電容ESR要在除以開關頻率fsw
        cap_rand_ESR = cap_rand[3] / fsw_rand
        #檢查ESR值
        Design_param_combination.append([fsw_rand , cap_rand[0] , cap_rand[1] , cap_rand[2] , cap_rand_ESR , cap_rand[4] , ind_rand[0] , ind_rand[1] , ind_rand[2]])

    #印出產生的模擬資料
    # for data in Design_param_combination:
    #     print(data)

    #再把8000筆資料傳到 ltspiceautorun_v_net 中產生模擬資料

    return Design_param_combination