import func_lib as fl
import select_com_bn_nn as sc
#test com_list()
list1 = ['a','b','c','d','e']
list2 = [[5,3],[4,2],[3,1],[2,0],[1,1]]
list3 = [1,2,3,4,5]
result = fl.com_list(list1,list3)
print(result)

#test find_ripple()
list_max = [[12.0529], [12.0459], [12.0711], [12.1268], [12.0167]]
list_min = [[11.7196], [11.9742], [11.8724], [11.9471], [11.9827]]
list_avg = [[11.9496], [11.9999], [11.9646], [12.0001], [12.0004]]
print(fl.find_ripple(list_max,list_min,list_avg))

#test max()
max_result = max(list3)
print(max_result)

print(round(80.55345,-1))

print(fl.cal_step(1,10,10))

result = sc.training_design_parameter(5)
print(result)



# num_values = 5
# f_range = {'f_min':20e3,'f_max':200e3}
# for fsw in fl.cal_step(f_range['f_min'],f_range['f_max'],num_values):
#     print(fsw)