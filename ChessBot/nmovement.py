import RPi.GPIO as GPIO
import time
 
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
"""
coil_A_1_pin = [3,12]
coil_A_2_pin = [7,15]
coil_B_1_pin = [5,13]
coil_B_2_pin = [11,16]
"""


coil_A_1_pin = [18,23]
coil_A_2_pin = [21,11]
coil_B_1_pin = [19,24]
coil_B_2_pin = [22,16]


# adjust if different
StepCount = 8
Seq = range(0, StepCount)
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
 
# GPIO.setup(enable_pin, GPIO.OUT)
for i in range(2):
    GPIO.setup(coil_A_1_pin[i], GPIO.OUT)
    GPIO.setup(coil_A_2_pin[i], GPIO.OUT)
    GPIO.setup(coil_B_1_pin[i], GPIO.OUT)
    GPIO.setup(coil_B_2_pin[i], GPIO.OUT)

# GPIO.output(enable_pin, 1)
 
def setStep(m1,w1, w2, w3, w4,m2,v1,v2,v3,v4):
    
    GPIO.output(coil_A_1_pin[m1], w1)
    GPIO.output(coil_A_1_pin[m2], v1)
    
    GPIO.output(coil_A_2_pin[m1], w2)
    GPIO.output(coil_A_2_pin[m2], v2)
    
    GPIO.output(coil_B_1_pin[m1], w3)
    GPIO.output(coil_B_1_pin[m2], v3)
    
    GPIO.output(coil_B_2_pin[m1], w4)
    GPIO.output(coil_B_2_pin[m2], v4)
 
def forward(m1,m2,delay, steps):
    if steps >= 0:
        for i in range(steps):
            for j in range(StepCount):
                setStep(m1,Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3],m2,Seq[7-j][0], Seq[7-j][1], Seq[7-j][2], Seq[7-j][3])
                time.sleep(delay)
    else:
        for i in range(-1*steps):
            for j in range(StepCount):
                setStep(m1,Seq[7-j][0], Seq[7-j][1], Seq[7-j][2], Seq[7-j][3],m2, Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
                time.sleep(delay)
#def exp(delay,j):
#    setStep(Seq1[j][0],Seq1[j][1],Seq1[j][2],Seq1[j][3])
    
if __name__ == '__main__':
    delay = raw_input("Time Delay (ms)?")
    while True:
        steps = raw_input("How many steps forward? ")
        forward(0,1,int(delay) / 1000.0, int(steps))
        #j = int(raw_input("Seq: "))
        #exp(delay,j)
        





