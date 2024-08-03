import pymysql
import math
import func_lib as fl
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(verbose=True)
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")
#電感與電容值範圍()
f_range = {'f_min':20e3,'f_max':200e3} #Hz
c_range = {'c_min':47,'c_max':726} #uF
l_range = {'l_min':100,'l_max':1000} #uH
#資料庫參數設定

db_settings_cap = {
    "host": DB_HOST,
    "port": int(DB_PORT),
    "user": DB_USER,
    "password": DB_PASSWORD,
    "db": "cap_database",
    "charset": "utf8"
}

db_settings_core = {
    "host": DB_HOST,
    "port": int(DB_PORT),
    "user": DB_USER,
    "password": DB_PASSWORD,
    "db": "core_database",
    "charset": "utf8"
}

try:
    #建立connection物件

    conn_cap = pymysql.connect(**db_settings_cap)
    conn_core= pymysql.connect(**db_settings_core)

    #建立cursor物件

    with conn_cap.cursor() as cursor_cap:

        # 查詢資料SQL語法
        command = "SELECT * FROM cap"

        #執行指令
        cursor_cap.execute(command)

        #取得所有資料
        result_cap = cursor_cap.fetchall()

        # maximum number of capacitors
        MP = 15
        # ESL factor 4.389*10^-11 電容值單位為uF
        Ksel = 4.389*1e-11
        # Dissipation factor
        dis_factor = 0.14
        # Cap lookup table
        cap_value = []
        cap_volume= []
        cap_ESR = []
        cap_ESL = []
        cap_part_no = []

        # 創建電容lookup table (不同串並聯的電感值\ESR\ESL\體積) Cmin = 20uF Cmax = 1000uF
        # single capacitor
    
        for i in range(len(result_cap)):
            
            #儲存part_no
            cap_part_no.append(result_cap[i][0])
            #計算電容值 單位uF
            cap_value.append(result_cap[i][2])

            #計算體積 單位cm^3
            volume_temp = result_cap[i][4]
            volume_index = volume_temp.index("X")
            diameter = float(volume_temp[:volume_index]) #電容直徑
            length = float(volume_temp[volume_index+1:]) #電容長度
            volume_result = (math.pi*(diameter/2) ** 2) * length / 1000 
            cap_volume.append(round(volume_result, 3))

            #計算ESR 尚未除以fs 電容值單位為uF
            ESR_temp=dis_factor/(2*math.pi*result_cap[i][2]*1e-6)
        
            #在choose_combination.py 除以開關頻率後再四捨五入
            cap_ESR.append(ESR_temp)

            #計算ESL
            cap_ESL.append(round(Ksel/(result_cap[i][2]*1e-6), 10))

        #series capacitor
        series_notation = '+'
        for i in range(len(result_cap)):
            for j in range(len(result_cap)-i):
                #儲存part_no "+":series
                cap_part_no_temp = result_cap[i][0] + series_notation + result_cap[i+j][0]
                cap_part_no.append(cap_part_no_temp)
                #計算電容值 單位uF
                cap_value_temp = (result_cap[i][2] * result_cap[i+j][2]) / (result_cap[i][2] + result_cap[i+j][2])
                cap_value.append(cap_value_temp)

                #計算體積 單位cm^3
                volume_result = fl.cal_cap_volume(result_cap[i][4]) + fl.cal_cap_volume(result_cap[i+j][4])
                cap_volume.append(round(volume_result, 3))

                #計算ESR 尚未除以fs 電容值單位為uF
                ESR_temp=dis_factor/(2*math.pi*cap_value_temp*1e-6)
            
                #在choose_combination.py 除以開關頻率後再四捨五入
                cap_ESR.append(ESR_temp)

                #計算ESL
                cap_ESL.append(round(Ksel/(cap_value_temp*1e-6), 10))

        #parralle capacitor
        parralle_notation = '/'
        for i in range(len(result_cap)):
            for j in range(len(result_cap)-i):
                #儲存part_no "/":parralle
                cap_part_no_temp = result_cap[i][0] + parralle_notation + result_cap[i+j][0]
                cap_part_no.append(cap_part_no_temp)
                #計算電容值 單位uF
                cap_value_temp = (result_cap[i][2] + result_cap[i+j][2])
                cap_value.append(cap_value_temp)

                #計算體積 單位cm^3
                volume_result = fl.cal_cap_volume(result_cap[i][4]) + fl.cal_cap_volume(result_cap[i+j][4])
                cap_volume.append(round(volume_result, 3))

                #計算ESR 尚未除以fs 電容值單位為uF
                ESR_temp=dis_factor/(2*math.pi*cap_value_temp*1e-6)
            
                #在choose_combination.py 除以開關頻率後再四捨五入
                cap_ESR.append(ESR_temp)

                #計算ESL
                cap_ESL.append(round(Ksel/(cap_value_temp*1e-6), 10))
        #parralle and series
        for k in range(len(result_cap)):
            for i in range(len(result_cap)):
                for j in range(len(result_cap)-i):
                    #儲存part_no "/":parralle
                    cap_part_no_temp = (result_cap[i][0] + series_notation + result_cap[i+j][0] 
                                        + parralle_notation + result_cap[k][0])
                    cap_part_no.append(cap_part_no_temp)
                    #計算電容值 單位uF
                    cap_value_temp = ((result_cap[i][2] * result_cap[i+j][2]) / (result_cap[i][2] + result_cap[i+j][2])
                                      + result_cap[k][2])
                    cap_value.append(cap_value_temp)

                    #計算體積 單位cm^3
                    volume_result = (fl.cal_cap_volume(result_cap[i][4]) + fl.cal_cap_volume(result_cap[i+j][4])
                    + fl.cal_cap_volume(result_cap[k][4])) 
                    cap_volume.append(round(volume_result, 3))

                    #計算ESR 尚未除以fs 電容值單位為uF
                    ESR_temp=dis_factor/(2*math.pi*cap_value_temp*1e-6)
                
                    #在choose_combination.py 除以開關頻率後再四捨五入
                    cap_ESR.append(ESR_temp)

                    #計算ESL
                    cap_ESL.append(round(Ksel/(cap_value_temp*1e-6), 10))
        # for index, data in enumerate(result_cap):
        #     #data[0] part_no
        #     #data[1] ratevoltage
        #     #data[2] cap_value
        #     #data[3] dissipation_factor
        #     #data[4] case_size
        #     #data[5] esl_factor
        #     print(index,data)
        #產生 capacitor table [(part_no , value , volume , ESR , ESL)]
        cap_table = []
        for i in range(len(cap_part_no)):
            if cap_value[i] > c_range['c_min'] and cap_value[i] < c_range['c_max']:
                cap_table.append([cap_part_no[i], cap_value[i], cap_volume[i], cap_ESR[i], cap_ESL[i]])
            # print(cap_value[i], end=" ")   
            # print(cap_volume[i], end=" ") 
            # print(cap_ESR[i], end=" ")
            # print(cap_ESL[i], end=" ") 
            # print(cap_part_no[i], end=" ")
            # print()
            # print(cap_table[i])
        

    with conn_core.cursor() as cursor_core:
        # 查詢資料SQL語法
        command = "SELECT * FROM core"

        #執行指令
        cursor_core.execute(command)

        #取得所有資料
        result_core = cursor_core.fetchall()
        core_data = result_core
        #fill factor Ku
        Ku = 0.35
        #Nmax index 代表core
        N_max = [74,93,162,56]
        #L_max index 代表core unit:uH
        L_max = [509,1003,2519,144]
        L_min = [144,509,1003,30] 
        #for i in range(len(result_core)):
            #儲存part_no
            #計算area of wire mm^2 (使用論文資料)
            
            #計算每種磁心的最大圈數 (使用論文資料)
            #N_max_temp = Ku*(math.pi*(float(result_core[i][2]) ** 2))/(4 * Aw)
            #N_max.append(N_max_temp)

            #計算每種磁心的最大電感值
            #L_max_temp = float(result_core[i][8])*0.001/1000*(N_max[i] ** 2)
            #L_max.append(L_max_temp)
        #儲存電感值    
        ind_value = []
        #儲存RL值
        ind_RL = []
        #儲存part_no 
        ind_core = []
        #儲存電感繞線圈數 N
        ind_N = []
        #儲存電感體積 volume
        ind_volume = []
        #產生 inductor lookup table Lmin = 30uH Lmax = 2000uH
        for i in range(len(result_core)):
            N_min = int(round((L_min[i]/float(result_core[i][8])*1000)**0.5, 0))
            for j in range(N_min+1, N_max[i]+1):
                #儲存 core part_no
                ind_core.append(result_core[i][0])
                #計算電感值並儲存
                ind_value_temp = float(result_core[i][8])* ((j) ** 2)*1e-3 #AL (nH/T^2) * N^2 
                #電感值 四捨五入
                ind_value_temp = round(ind_value_temp, 2)
                ind_value.append(ind_value_temp)
                #計算RL 假設r(resistance of wire) = 2.14 * 10^-5 歐姆/mm
                ind_RL_temp = (j) * (float(result_core[i][1]) - float(result_core[i][2]) + 2 * float(result_core[i][3])) * 2.14 * 1e-5
                ind_RL_temp = round(ind_RL_temp, 6)
                ind_RL.append(ind_RL_temp)
                #儲存圈數 N
                ind_N.append(j)
                #計算電感體積 pi*(OD/2)^2*Ht (unit: cm^3)
                od_divide_2 = float(result_core[i][1])/2
                ind_volume_temp = (3.14*od_divide_2*od_divide_2*float(result_core[i][3]))/1000
                ind_volume.append(round(ind_volume_temp,3))

        #產生inductor lookup table (ind_core ,  ind_value , ind_RL , ind_N, ind_volume)
        ind_table = []
        for i in range(len(ind_core)):
            #排除範圍外的電感值
            if ind_value[i] > l_range['l_min'] and ind_value[i] < l_range['l_max']:
                ind_table.append([ind_core[i] , ind_value[i] , ind_RL[i], ind_N[i], ind_volume[i]])
            # print(ind_core[i], end = " ")
            # print(ind_value[i], end = " ")
            # print(ind_RL[i], end = " ")
            # print()

        # for ind_data in ind_table:
        #     print(ind_data)
        
        # for data in result_core:
        #     #[0] part_no
        #     #[1] OD(mm)
        #     #[2] ID(mm)
        #     #[3] Ht(mm)
        #     #[4] L(cm)
        #     #[5] Ae
        #     #[6] A
        #     #[7] Volume
        #     #[8] AL(nH/T^2)
        #     print()
        #print(N_max)
        #print(L_max)   
        
except Exception as ex:
    print(ex)

def cap_table_return():
    return cap_table

def ind_table_return():
    return ind_table
