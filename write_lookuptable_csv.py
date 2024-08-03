import gen_lookup_table as lookuptable
import csv
#capacitor table [(part_no , value , volume , ESR , ESL)]
cap_lookup_table=lookuptable.cap_table_return()

#inductor lookup table (ind_core ,  ind_value , ind_RL , ind_N, ind_volume)
ind_lookup_table=lookuptable.ind_table_return()

#cap table csv (ESR尚未除以開關頻率)
with open('cap_table.csv', 'w', newline='') as csvfile_cap_table:
    fieldnames = ['part_no', 'value', 'volume', 'ESR', 'ESL']
    writer = csv.DictWriter(csvfile_cap_table, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(cap_lookup_table)):
        writer.writerow({'part_no': cap_lookup_table[i][0], 'value': cap_lookup_table[i][1], 'volume': cap_lookup_table[i][2], 'ESR': cap_lookup_table[i][3], 'ESL': cap_lookup_table[i][4]})

#ind table csv
with open('ind_table.csv', 'w', newline='') as csvfile_ind_table:
    fieldnames = ['ind_core', 'ind_value', 'ind_RL', 'ind_N', 'ind_volume']
    writer = csv.DictWriter(csvfile_ind_table, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(ind_lookup_table)):
        writer.writerow({'ind_core': ind_lookup_table[i][0], 'ind_value': ind_lookup_table[i][1], 'ind_RL': ind_lookup_table[i][2], 'ind_N': ind_lookup_table[i][3], 'ind_volume': ind_lookup_table[i][4]})

