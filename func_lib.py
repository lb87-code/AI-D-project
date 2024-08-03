import math
import pymysql
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from pathlib import Path

def com_list(list1, list2):
    result = []
    #two list must be same length
    if len(list1) != len(list2):
        print("two list length must be equal!")
        return 1
    # list1 = ['a','b','c','d','e']
    # list2 = [5,4,3,2,1]
    # result = [['a', 3], ['b', 3], ['c', 3], ['d', 3], ['e', 3]]
    else:
        for i in range(len(list1)):
            result.append([list1[i], list2[i]])
    
    return result

def find_ripple(max_val,min_val,avg_val):
    result = []
    if len(max_val) == len(min_val) == len(avg_val):
        for i in range(len(avg_val)):
            temp = (max_val[i][0]-min_val[i][0])/avg_val[i][0] *100
            result.append(temp)
    else:
        print("lenght error of ripple data!")
        return 1
    return result

def find_eff(pin,pout):
    result = []
    if len(pin) == len(pout):
        for i in range(len(pout)):
            temp = pout[i][0]/pin[i][0]*100
            result.append(temp)
    else:
        print("lenght error of ripple data!")
        return 1
    return result

def cal_cap_volume(cap_volume):
    volume_temp = cap_volume
    volume_index = volume_temp.index("X")
    diameter = float(volume_temp[:volume_index]) #電容直徑
    length = float(volume_temp[volume_index+1:]) #電容長度
    volume_result = (math.pi*(diameter/2) ** 2) * length / 1000 #unit cm^3
    return volume_result


def cal_step(start,end,num_values):
    step = (end - start) / (num_values - 1)
    values = [start + i * step for i in range(num_values)]
    return values

def cal_ind_turns(ind,al):
    N = (ind/al)**0.5
    return N

def access_core_data():
    load_dotenv(verbose=True)
    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT")
    
    db_settings_core = {
    "host": DB_HOST,
    "port": int(DB_PORT),
    "user": DB_USER,
    "password": DB_PASSWORD,
    "db": "core_database",
    "charset": "utf8"
    }
    try:
        conn_core= pymysql.connect(**db_settings_core)
        with conn_core.cursor() as cursor_core:
            # 查詢資料SQL語法
            command = "SELECT * FROM core"

            #執行指令
            cursor_core.execute(command)

            #取得所有資料
            result_core = cursor_core.fetchall()

    except Exception as ex:
        print(ex)

    return result_core

# print(history_sw_h.history.keys()) #查詢key值
#plot trian set and validation set a.accurate b.loss
def print_history(history,title,x,y):
  # plt.plot(history.history['accuracy'])
  # plt.plot(history.history['val_accuracy'])
  plt.plot(history.history['loss'], label = 'train_loss')
  plt.plot(history.history['val_loss'], label = 'val_loss')
  plt.title(title)
  plt.xlabel('epoch')
  plt.legend()
  x_major_locator = plt.MultipleLocator(x)
  y_major_locator = plt.MultipleLocator(y)
  ax = plt.gca()
  ax.xaxis.set_major_locator(x_major_locator)
  ax.yaxis.set_major_locator(y_major_locator)
  plt.show()

def cap_esr_cal(fsw,cap,dis_factor):
    cap_esr = dis_factor/2.0/math.pi/fsw/cap
    return cap_esr
  
def cap_esl_cal(cap,ksel):
    cap_esl = ksel/cap
    return cap_esl
  
def ind_esr_cal(ind,l_range):
    #計算電感寄生參數
    # assume r(resistance of wire) = 2.14 * 10^-5 歐姆/mm
    r = 2.14e-5
    core_data = access_core_data()
    if  ind <= 144e-6 and ind >= l_range['l_min']:
        i = 3
        N = cal_ind_turns(ind,float(core_data[i][8])*1e-9)
        OD = float(core_data[i][1])
        ID = float(core_data[i][2])
        H = float(core_data[i][3])
        ind_esr = N*(OD-ID+2*H)*r
    elif ind <= 509e-6 and ind > 144e-6:
        i = 0
        N = cal_ind_turns(ind,float(core_data[i][8])*1e-9)
        OD = float(core_data[i][1])
        ID = float(core_data[i][2])
        H = float(core_data[i][3])
        ind_esr = N*(OD-ID+2*H)*r
    elif ind <= 1003e-6 and ind > 509e-6:
        i = 1
        N = cal_ind_turns(ind,float(core_data[i][8])*1e-9)
        OD = float(core_data[i][1])
        ID = float(core_data[i][2])
        H = float(core_data[i][3])
        ind_esr = N*(OD-ID+2*H)*r
    elif ind <= 2519e-6 and ind > 1003e-6:
        i = 2
        N = cal_ind_turns(ind,float(core_data[i][8])*1e-9)
        OD = float(core_data[i][1])
        ID = float(core_data[i][2])
        H = float(core_data[i][3])
        ind_esr = N*(OD-ID+2*H)*r
    else:
        print("warnning! inductance out of range")
    return ind_esr 