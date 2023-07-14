#!/usr/bin/env python
# -- encoding: UTF-8 --
import rospy
import qi
import argparse
import sys
import almath
import math
import motion
import time
import os
from naoqi import ALProxy


from std_msgs.msg import Int32, Float32MultiArray

session = qi.Session()
session.connect("tcp://192.168.0.103")

animated_speech_service = session.service("ALAnimatedSpeech")
behavior_service = session.service("ALBehaviorManager")

def eeg_data_callback(msg):
    # Obtener el número recibido del mensaje
    i = msg.data

    # Opcion 1
    if i == 1:
	print("Funcionalidad 1")
        animated_speech_service.say("\\vct=100\\El proyecto de grado en el que me desempeño es un sistema BCI, en el cual se realiza la recepción de ondas cerebrales mediante la diadema MindWave Mobile 2, se procesan mediante un algoritmo de inteligencia artificial y finalmente se me dan los parámetros para el funcionamiento.")
    # Opcion 2
    elif i == 2:
        print("Funcionalidad 2")
        behavior_service.runBehavior("animations/Stand/Gestures/Hey_6")
    # Opcion 3
    elif i == 3:
        print("Funcionalidad 3")
        motion_service = session.service("ALMotion")
        joint_names = ["RShoulderPitch", "RHand", "RElbowYaw", "RElbowRoll"]
        handshake_angles = [0.4, 1.0, 0.0,0.0]#Saludo Mano
        motion_service.angleInterpolationWithSpeed(joint_names, handshake_angles,0.1)
def eeg_data_subscriber():

    rospy.init_node('eeg_data_subscriber', anonymous=True)

    rospy.Subscriber('parpadeos', Int32, eeg_data_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        eeg_data_subscriber()
    except rospy.ROSInterruptException:
        pass

