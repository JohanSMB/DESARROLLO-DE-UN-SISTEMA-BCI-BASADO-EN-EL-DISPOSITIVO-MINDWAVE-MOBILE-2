# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.models import Sequential

data = pd.read_csv('Dataset1.csv')
input_data = data['EEG'].values

data2 = pd.read_csv('Dataset2.csv')
input_data2 = data2['EEG'].values

data4 = pd.read_csv('Dataset4.csv')
input_data4 = data4['EEG'].values

data6 = pd.read_csv('Dataset6.csv')
input_data6 = data6['EEG'].values


input_grup = [input_data6[i:20+i] for i in range(0, 80)]+[input_data6[i:20+i] for i in range(100, 180)]+[input_data6[i:20+i] for i in range(200, 280)]+[input_data6[i:20+i] for i in range(300, 380)]+[input_data6[i:20+i] for i in range(400, 480)]+[input_data6[i:20+i] for i in range(500, 580)]+[input_data6[i:20+i] for i in range(600, 680)]+[input_data6[i:20+i] for i in range(700, 780)]+[input_data6[i:20+i] for i in range(800, 880)]+[input_data6[i:20+i] for i in range(900, 980)]+[input_data6[i:20+i] for i in range(1000, 1080)]+[input_data6[i:20+i] for i in range(1100, 1180)]+[input_data6[i:20+i] for i in range(1200, 1280)]+[input_data6[i:20+i] for i in range(1300, 1380)]+[input_data6[i:20+i] for i in range(1400, 1480)]+[input_data6[i:20+i] for i in range(1500, 1580)]+[input_data6[i:20+i] for i in range(1600, 1680)]+[input_data6[i:20+i] for i in range(1700, 1780)]+[input_data6[i:20+i] for i in range(1800, 1880)]+[input_data6[i:20+i] for i in range(1900, 1980)]+[input_data6[i:20+i] for i in range(2000, 2080)]+[input_data6[i:20+i] for i in range(2100, 2180)]+[input_data6[i:20+i] for i in range(2200, 2280)]+[input_data6[i:20+i] for i in range(2300, 2380)]+[input_data6[i:20+i] for i in range(2400, 2480)]+[input_data6[i:20+i] for i in range(2500, 2580)]+[input_data6[i:20+i] for i in range(2600, 2680)]+[input_data6[i:20+i] for i in range(2700, 2780)]+[input_data6[i:20+i] for i in range(2800, 2880)]+[input_data6[i:20+i] for i in range(2900, 2980)]+[input_data6[i:20+i] for i in range(3000, 3080)]+[input_data6[i:20+i] for i in range(3100, 3180)]+[input_data6[i:20+i] for i in range(3200, 3280)]+[input_data6[i:20+i] for i in range(3300, 3380)]+[input_data6[i:20+i] for i in range(3400, 3480)]+[input_data6[i:20+i] for i in range(3500, 3580)]+[input_data6[i:20+i] for i in range(3600, 3680)]+[input_data6[i:20+i] for i in range(3700, 3780)]+[input_data6[i:20+i] for i in range(3800, 3880)]+[input_data6[i:20+i] for i in range(3900, 3980)]+[input_data6[i:20+i] for i in range(4000, 4080)]+[input_data6[i:20+i] for i in range(4100, 4180)]+[input_data6[i:20+i] for i in range(4200, 4280)]+[input_data6[i:20+i] for i in range(4300, 4380)]+[input_data6[i:20+i] for i in range(4400, 4480)]+[input_data6[i:20+i] for i in range(4500, 4580)]+[input_data6[i:20+i] for i in range(4600, 4680)]+[input_data6[i:20+i] for i in range(4800, 4880)]+[input_data6[i:20+i] for i in range(4900, 4980)]+[input_data6[i:20+i] for i in range(5000, 5080)]+[input_data6[i:20+i] for i in range(5100, 5180)]+[input_data6[i:20+i] for i in range(5200, 5280)]+[input_data6[i:20+i] for i in range(5300, 5380)]+[input_data6[i:20+i] for i in range(5400, 5480)]+[input_data6[i:20+i] for i in range(5500, 5580)]+[input_data6[i:20+i] for i in range(5600, 5680)]+[input_data6[i:20+i] for i in range(5700, 5780)]+[input_data6[i:20+i] for i in range(5800, 5880)]+[input_data6[i:20+i] for i in range(5900, 5880)]+[input_data6[i:20+i] for i in range(6000, 6080)]+[input_data6[i:20+i] for i in range(6100, 6180)]+[input_data4[i:20+i] for i in range(150, 230)]+[input_data4[i:20+i] for i in range(425, 505)]+[input_data4[i:20+i] for i in range(740, 820)]+[input_data4[i:20+i] for i in range(900, 980)]+[input_data4[i:20+i] for i in range(1050, 1130)]+[input_data4[i:20+i] for i in range(1370, 1450)]+[input_data4[i:20+i] for i in range(1700, 1780)]+[input_data4[i:20+i] for i in range(2000, 2080)]+[input_data4[i:20+i] for i in range(2300, 2380)]+[input_data4[i:20+i] for i in range(2580, 2660)]+[input_data4[i:20+i] for i in range(2825, 2905)]+[input_data4[i:20+i] for i in range(3050, 3130)]+[input_data4[i:20+i] for i in range(3300, 3380)]+[input_data4[i:20+i] for i in range(3550, 3630)]+[input_data4[i:20+i] for i in range(3750, 3830)]+[input_data4[i:20+i] for i in range(3980, 4060)]+[input_data4[i:20+i] for i in range(4250, 4330)]+[input_data4[i:20+i] for i in range(4475, 4555)]+[input_data4[i:20+i] for i in range(4675, 4755)]+[input_data2[i:20+i] for i in range(350, 430)]+[input_data2[i:20+i] for i in range(570, 650)]+[input_data2[i:20+i] for i in range(800, 880)]+[input_data2[i:20+i] for i in range(1000, 1080)]+[input_data2[i:20+i] for i in range(1225, 1305)]+[input_data2[i:20+i] for i in range(1425, 1505)]+[input_data2[i:20+i] for i in range(1640, 1720)]+[input_data2[i:20+i] for i in range(1850, 1930)]+[input_data2[i:20+i] for i in range(2050, 2130)]+[input_data2[i:20+i] for i in range(2275, 2355)]+[input_data2[i:20+i] for i in range(2460, 2540)]+[input_data2[i:20+i] for i in range(2660, 2740)]+[input_data2[i:20+i] for i in range(2850, 2930)]+[input_data2[i:20+i] for i in range(3050, 3130)]+[input_data2[i:20+i] for i in range(3240, 3320)]+[input_data2[i:20+i] for i in range(3450, 3530)]+[input_data2[i:20+i] for i in range(3640, 3720)]+[input_data2[i:20+i] for i in range(3850, 3930)]+[input_data2[i:20+i] for i in range(4075, 4155)]+[input_data2[i:20+i] for i in range(4300, 4380)]+[input_data2[i:20+i] for i in range(4570, 4650)]+[input_data[j:20+j] for j in range(400, 480)]+[input_data[j:20+j] for j in range(660, 740)]+[input_data[j:20+j] for j in range(900, 980)]+[input_data[j:20+j] for j in range(1100, 1180)]+[input_data[j:20+j] for j in range(1300, 1380)]+[input_data[j:20+j] for j in range(1550, 1630)]+[input_data[j:20+j] for j in range(1750, 1830)]+[input_data[j:20+j] for j in range(1940, 2020)]+[input_data[j:20+j] for j in range(2140, 2220)]+[input_data[j:20+j] for j in range(2345, 2425)]+[input_data[j:20+j] for j in range(2525, 2605)]+[input_data[j:20+j] for j in range(2710, 2790)]+[input_data[j:20+j] for j in range(2910, 2990)]+[input_data[j:20+j] for j in range(3300, 3380)]+[input_data[j:20+j] for j in range(3480, 3560)]+[input_data[j:20+j] for j in range(3680, 3760)]+[input_data[j:20+j] for j in range(3875, 3955)]+[input_data[j:20+j] for j in range(4070, 4150)]+[input_data[i:20+i] for i in range(4270, 4350)]+[input_data[i:20+i] for i in range(4490, 4570)]
print(len(input_grup))

output_data = np.zeros((9600, 2))
output_data[0:4800, 0] = 1  # 0 parpadeos
output_data[4800:9600, 1] = 1  # 1 parpadeos

print ("-------------", len(output_data))

input_data = np.array(input_grup).reshape(9600, 20, 1)
input_data = input_data.astype(float)

model = Sequential()
model.add(LSTM(128, input_shape=(20, 1), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(64))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(input_data, output_data, epochs=50, batch_size=10)

model.save('Modelo_Parpadeos128_64_50.h5')


