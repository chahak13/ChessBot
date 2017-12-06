import RPi.GPIO as GPIO
import time
 
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
coil_A_1_pin = 3
coil_A_2_pin = 7
coil_B_1_pin = 5
coil_B_2_pin = 11 
 
# adjust if different
StepCount = 8
Seq = range(0, StepCount)
#
Seq[0] = [1,0,0,0]
Seq[1] = [1,1,0,0]
#
Seq[2] = [0,1,0,0]
Seq[3] = [0,1,1,0]
#
Seq[4] = [0,0,1,0]
Seq[5] = [0,0,1,1]
#
Seq[6] = [0,0,0,1]
Seq[7] = [1,0,0,1]

#for reverse
Seq1 = range(0,StepCount)
Seq1[0] = [0,0,1,1]
Seq1[1] = [0,1,0,0]
Seq1[2] = [1,1,0,0]
Seq1[3] = [0,0,0,1]
#Seq1[4] = [1,1,0,0]
#Seq1[5] = [0,1,1,0]
#Seq1[6] = [0,0,1,1]
#Seq1[7] = [1,0,0,1]
#Seq1[8] = [0,1,0,1]
#Seq1[9] = [1,0,1,0]
 
# GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(coil_A_1_pin, GPIO.OUT)
GPIO.setup(coil_A_2_pin, GPIO.OUT)
GPIO.setup(coil_B_1_pin, GPIO.OUT)
GPIO.setup(coil_B_2_pin, GPIO.OUT)
 
# GPIO.output(enable_pin, 1)
 
def setStep(w1, w2, w3, w4):
    GPIO.output(coil_A_1_pin, w1)
    GPIO.output(coil_A_2_pin, w2)
    GPIO.output(coil_B_1_pin, w3)
    GPIO.output(coil_B_2_pin, w4)
 
def forward(delay, steps):
    if steps >= 0:
        for i in range(steps):
            for j in range(StepCount):
                setStep(Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
                time.sleep(delay)
    else:
        for i in range(-1*steps):
            for j in range(StepCount):
                setStep(Seq[7-j][0], Seq[7-j][1], Seq[7-j][2], Seq[7-j][3])
                time.sleep(delay)
#def exp(delay,j):
#    setStep(Seq1[j][0],Seq1[j][1],Seq1[j][2],Seq1[j][3])
    
if __name__ == '__main__':
    delay = raw_input("Time Delay (ms)?")
    while True:
        steps = raw_input("How many steps forward? ")
        forward(int(delay) / 1000.0, int(steps))
        #j = int(raw_input("Seq: "))
        #exp(delay,j)
        







