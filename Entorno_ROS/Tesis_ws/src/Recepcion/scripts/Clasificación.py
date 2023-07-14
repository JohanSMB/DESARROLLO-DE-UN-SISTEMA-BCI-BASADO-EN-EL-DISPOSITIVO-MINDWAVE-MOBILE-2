#!/usr/bin/env python
# -- coding: utf-8 --

import rospy
from std_msgs.msg import Int32, Bool
from std_msgs.msg import Float32MultiArray
import os
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Ruta y nombre del archivo del modelo

ruta_modelo = os.path.expanduser('~/Descargas/Modelo_Parpadeos128_64_50.h5')
# Cargar el modelo
model = load_model(ruta_modelo, compile=False)

# Declaración del publicador
publicador_parpadeos = rospy.Publisher('parpadeos', Int32, queue_size=10)

def eeg_callback(msg):
    input_data = np.array(msg.data)
    #print(len(input_data))
    mask = ((input_data >= 0) & (input_data <= 400)) | ((input_data >= -400) & (input_data <= 0))
    input_data = np.where(mask, 0, input_data)
    #print(".............",input_data)
    input_grup = [input_data[i:20+i] for i in range(len(input_data)-20+1)]
    input_data = np.array(input_grup).reshape(-1, 20, 1)

    predictions = model.predict(input_data)
    blinks = np.argmax(predictions, axis=1) == 1

    blink_count = 0
    total_blinks = 0
    in_blink = False  # Variable para rastrear si estamos dentro de un parpadeo

    for blink in blinks:
        if blink and not in_blink:
            in_blink = True
            blink_count = 1
        elif blink and in_blink:
            blink_count += 1
        elif not blink and in_blink:
            in_blink = False
            if blink_count >= 24:#24
                total_blinks += 1
    rospy.loginfo("Número total de parpadeos: %d", total_blinks)
    #rospy.loginfo (blinks)

    publicador_parpadeos.publish(total_blinks)

def eeg_subscriber():
    rospy.init_node('eeg_subscriber')  # Inicializar el nodo de ROS
    rospy.Subscriber('eeg_data', Float32MultiArray, eeg_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        eeg_subscriber()
    except rospy.ROSInterruptException:
        pass

