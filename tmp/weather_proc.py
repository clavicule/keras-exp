import argparse
from datetime import datetime
import numpy as np

# timeslot indexing funtion
def get_time_index(timestamp):
    day = int(timestamp.date().day) - 1
    slot = int((timestamp.time().hour * 3600 + timestamp.time().minute * 60 + timestamp.time().second) / 600)
    return day * 144 + slot

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weather", required=True, help="Path to the weather data file")
ap.add_argument("-o", "--output", required=True, help="Path to the output file")
args = vars(ap.parse_args())

total_timeslots = 19 * 144
weather_dataset = np.zeros((total_timeslots, 11), dtype="float")

print('reading weather')
weather_file = open(args['weather'], 'r')
for line in weather_file:
    weather_data = line.split('\t')
    time_key = get_time_index(datetime.strptime(weather_data[0].strip(), '%Y-%m-%d %H:%M:%S'))

    if time_key > total_timeslots:
        continue

    climate = int(weather_data[1].strip())
    temperature = float(weather_data[2].strip())
    pollution = float(weather_data[3].strip())

    weather_dataset[time_key][climate - 1] += 1.
    weather_dataset[time_key][9] += temperature
    weather_dataset[time_key][10] += pollution
weather_file.close()


count = np.sum(weather_dataset[:, 0:9], axis=1)
count[ count == 0 ] = 1.;

weather_dataset[:, 9] = weather_dataset[:, 9] / count
weather_dataset[:, 10] = weather_dataset[:, 10] / count
np.savetxt(args["output"], weather_dataset, delimiter=',', fmt='%f')
