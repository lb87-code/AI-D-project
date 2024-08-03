#讀取csv檔案並且把內容存成list
import pandas as pd
import csv
import math

#讀取cap_table 並且儲存[[cap_v1, cap_v2 ...] [cap_volume1, cap_volume2 ...] [part_no_1, part_no_2 ...]]
#讀取下來的資料為pandas dataframe格式 (encode 方式更改為 latin-1)
cap_table = pd.read_csv(r"C:\Users\samuel\Desktop\mywork\ltspiceauto-code\cap_table.csv", names=["part_no","value","volume","ESR","ESL"])
#選擇存成list格式的特定行
selected_colums = ['value','volume','part_no']
#將選擇的行存到list中
list_of_selected_columns = [cap_table[column].tolist() for column in selected_colums]

# print(list_of_selected_columns)
#去除標題
list_of_value = list_of_selected_columns[0]
list_of_value.pop(0)
list_of_volume = list_of_selected_columns[1]
list_of_volume.pop(0)
list_of_part_no = list_of_selected_columns[2]
list_of_part_no.pop(0)

cap = []
float_cap_list = [float(item)*1e-6 for item in list_of_value]
print(float_cap_list)


#讀取ind_table 並且儲存[[ind_v1, ind_v2 ...] [ind_volume1, ind_volume2 ...] [part_no_1, part_no_2 ...] [ind_n1, ind_n2 ...]]
#讀取下來的資料為pandas dataframe格式
ind_table = pd.read_csv(r"C:\Users\samuel\Desktop\mywork\ltspiceauto-code\ind_table.csv", names=["part_no","value","ESR","ind_N","volume"])
#選擇存成list格式的特定行
selected_colums = ['value','volume','part_no','ind_N']
#將選擇的行存到list中
list_of_selected_columns = [ind_table[column].tolist() for column in selected_colums]

# print(list_of_selected_columns)
#去除標題
list_of_value = list_of_selected_columns[0]
list_of_value.pop(0)
list_of_volume = list_of_selected_columns[1]
list_of_volume.pop(0)
list_of_part_no = list_of_selected_columns[2]
list_of_part_no.pop(0)
list_of_ind_N = list_of_selected_columns[3]
list_of_ind_N.pop(0)



