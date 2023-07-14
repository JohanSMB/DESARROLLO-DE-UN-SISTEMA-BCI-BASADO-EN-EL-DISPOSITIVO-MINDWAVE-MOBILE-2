#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NeuroPy import NeuroPy
import time
import rospy
from std_msgs.msg import Float32MultiArray,MultiArrayLayout, MultiArrayDimension
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


os.system("sudo chmod 666 /dev/rfcomm0")

neuropy = NeuroPy("/dev/rfcomm0")

def publish_eeg_data():
    rospy.init_node('eeg_publisher')
    pub = rospy.Publisher('eeg_data', Float32MultiArray, queue_size=10)
    eeg_values = []  
    print("Presiona ENTER para iniciar...")
    raw_input()
      
    neuropy.start()
    start_time = time.time()
    elapsed_time = 0
    while elapsed_time < 6:
         current_time = time.time()
         elapsed_time = current_time - start_time
         eeg_value = neuropy.rawValue
         eeg_values.append(eeg_value)  # Agregar el valor de EEG a la lista
         time.sleep(0.01)
    neuropy.stop()    
   
    eeg_msg = Float32MultiArray(layout=MultiArrayLayout(dim=[MultiArrayDimension(label="eeg_data", size=len(eeg_values), stride=1)], data_offset=0), data=eeg_values)

    pub.publish(eeg_msg)

    print(len(eeg_values))

    plt.plot(eeg_values) 
    plt.xlabel('t')
    plt.ylabel('Datos Normalizados')
    plt.title('Gráfico de Datos Normalizados')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    try:
	eeg_values = publish_eeg_data()
	
    except rospy.ROSInterruptException:
        pass
