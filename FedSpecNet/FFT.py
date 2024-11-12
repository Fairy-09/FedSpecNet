import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
np.random.seed(0)
timestamps = ["6/26/2018 6:00:00 PM", "6/26/2018 6:05:00 PM", "6/26/2018 6:10:00 PM"]
for i in range(997):
    ts = datetime.datetime.strptime(timestamps[-1], "%m/%d/%Y %I:%M:%S %p") + datetime.timedelta(minutes=5)
    timestamps.append(ts.strftime("%m/%d/%Y %I:%M:%S %p"))

data = pd.read_csv('/home/liujian/project/Time-Series-Library-main/dataset-new/house1_5min_KWh.csv', parse_dates=['date'])
data.set_index('date', inplace=True)
values = data['value'].values[:8400]
datetime_objects = [datetime.datetime.strptime(ts, "%m/%d/%Y %I:%M:%S %p") for ts in timestamps]
time_deltas = [(dt - datetime_objects[0]).total_seconds() / 60 for dt in datetime_objects]

Fs = 1 / (time_deltas[1] - time_deltas[0])
T = 1 / Fs
Y = np.fft.fft(values)
Y[0]=0
f = np.fft.fftfreq(len(values), d=T)

plt.figure(figsize=(10, 5))
plt.subplots_adjust(left=0.05, bottom=0.125, right=0.99, top=0.94)
plt.title("House1_frequency_spectrum",fontsize=20)
positive_f = f[f > 0]
positive_Y = Y[:len(positive_f)]
amplitudes = np.abs(positive_Y)
plt.plot(positive_f, amplitudes)
plt.xlabel('Frequency (Hz)',fontsize=12)
plt.ylabel('Amplitude',fontsize=12)

plt.show()


