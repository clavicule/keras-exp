import argparse
from datetime import datetime
import numpy as np
from progress.bar import Bar

# timeslot indexing funtion
def get_time_index(timestamp):
    day = int(timestamp.date().day)
    slot = int((timestamp.time().hour * 3600 + timestamp.time().minute * 60 + timestamp.time().second) / 600)
    return day * 144 + slot


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--district", required=True, help="Path to the district ID file")
ap.add_argument("-o", "--order", required=True, help="Path to the order input data file")
ap.add_argument("-ov", "--order_val", required=True, help="Path to the order validating data file")
ap.add_argument("-t", "--training", help="Path to the output training file")
ap.add_argument("-v", "--validating", help="Path to the output validating file")
args = vars(ap.parse_args())

# get district ID mapping
print('reading district...')
district_mapping = {}
district_file = open(args['district'], 'r')
for line in district_file:
    district_id = line.split('\t')
    district = district_id.pop(0)
    district_mapping[district] = int(district_id[0].strip()) - 1
district_file.close()

total_timeslots = 31 * 144
no_district = len(district_mapping)
print('total number of district: {}'.format(no_district))
# data hold per timestep per district --> destination district count, price, gap
init_dataset = np.zeros((total_timeslots, no_district, no_district + 2), dtype="float")

# get input data
print('reading orders (training)')
min_timestep = total_timeslots
max_timestep = 0
order_file = open(args['order'], 'r')
for line in order_file:
    order_data = line.split('\t')
    time_key = get_time_index(datetime.strptime(order_data.pop().strip(), '%Y-%m-%d %H:%M:%S'))
    min_timestep = min(time_key, min_timestep)
    max_timestep = max(time_key, max_timestep)
    origin_district = order_data[3].strip()
    destination_district = order_data[4].strip()
    if origin_district not in district_mapping or destination_district not in district_mapping:
        continue
    init_dataset[time_key][district_mapping[origin_district]][district_mapping[destination_district]] += 1
    init_dataset[time_key][district_mapping[origin_district]][no_district] += float(order_data[5].strip()) # price
    if order_data[1].strip() == 'NULL':
        init_dataset[time_key][district_mapping[origin_district]][no_district + 1] += 1 # gap
order_file.close()

print('reading orders (validating)')
val_order_file = open(args['order_val'], 'r')
for line in val_order_file:
    order_data = line.split('\t')
    time_key = get_time_index(datetime.strptime(order_data.pop().strip(), '%Y-%m-%d %H:%M:%S'))
    origin_district = order_data[3].strip()
    destination_district = order_data[4].strip()
    if origin_district not in district_mapping or destination_district not in district_mapping:
        continue
    init_dataset[time_key][district_mapping[origin_district]][district_mapping[destination_district]] += 1
    init_dataset[time_key][district_mapping[origin_district]][no_district] += float(order_data[5].strip()) # price
    if order_data[1].strip() == 'NULL':
        init_dataset[time_key][district_mapping[origin_district]][no_district + 1] += 1 # gap
val_order_file.close()

train_dataset = init_dataset[min_timestep:max_timestep+1]
# train_dataset = train_dataset.reshape(train_dataset.shape[0] * train_dataset.shape[1], train_dataset.shape[2])
print(train_dataset.shape)

val_dataset = init_dataset[max_timestep+1:total_timeslots]
# val_dataset = val_dataset.reshape(val_dataset.shape[0] * val_dataset.shape[1], val_dataset.shape[2])
print(val_dataset.shape)

print('dumping training data to file')
bar = Bar('Saving...', max=train_dataset.shape[0] * train_dataset.shape[1] * train_dataset.shape[2])
training_file = open(args['training'], 'w')
for j in range(train_dataset.shape[1]):
    for i in range(train_dataset.shape[0]):
        no_order = 0
        gap = train_dataset[i][j][no_district + 1]
        training_file.write(str(j+1) + ',')
        for k in range(no_district):
            bar.next()
            no_order += train_dataset[i][j][k]
            training_file.write(str(train_dataset[i][j][k]) + ',')
        completed_order = no_order - gap
        if completed_order == 0:
            avg_price = 0
        else:
            avg_price = train_dataset[i][j][no_district] / completed_order
        training_file.write(str(avg_price) + ',' + str(train_dataset[i][j][no_district + 1]) + '\n')
bar.finish()
training_file.close()

print('dumping validating data to file')
bar = Bar('Saving...', max=val_dataset.shape[0] * val_dataset.shape[1] * val_dataset.shape[2])
val_file = open(args['validating'], 'w')
for j in range(val_dataset.shape[1]):
    for i in range(val_dataset.shape[0]):
        no_order = 0
        gap = val_dataset[i][j][no_district + 1]
        val_file.write(str(j + 1) + ',')
        for k in range(no_district):
            bar.next()
            no_order += val_dataset[i][j][k]
            val_file.write(str(val_dataset[i][j][k]) + ',')
        completed_order = no_order - gap
        if completed_order == 0:
            avg_price = 0
        else:
            avg_price = val_dataset[i][j][no_district] / completed_order
        val_file.write(str(avg_price) + ',' + str(val_dataset[i][j][no_district + 1]) + ',')
        if no_order == 0:
            val_file.write(str('0') + '\n')
        else:
            val_file.write(str('1') + '\n')
bar.finish()
val_file.close()
