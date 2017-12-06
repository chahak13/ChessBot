import RPi.GPIO as GPIO                                 ## Import GPIO Library.
import time                                 ## Import ‘time’ library for a delay.

GPIO.setmode(GPIO.BOARD)                    ## Use BOARD pin numbering.
GPIO.setup(26, GPIO.OUT)                    ## set output.

pwm=GPIO.PWM(26,50)                        ## PWM Frequency
pwm.start(7)

angle1=90
duty1= 7          ## Angle To Duty cycle  Conversion

angle2=180
duty2= 11
time.sleep(1)
ck=0
while ck<=0:
     pwm.ChangeDutyCycle(duty2)
     time.sleep(10)
     pwm.ChangeDutyCycle(duty1)
     time.sleep(1)
     ck=ck+1
time.sleep(1)
GPIO.cleanup()
