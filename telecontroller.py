#import dronekit_sitl
#from dronekit import connect,VehicleMode
import time
from dronekit import connect
#connect to vehicle
#sitl = dronekit_sitl.start_default()
#connection_string = sitl.connection_string()
connection_string= '/dev/ttyACM0'
vehicle = connect(connection_string, wait_ready=True)

print "Read channels individually:"
print " Ch1: %s" % vehicle.channels['1']
print " Ch2: %s" % vehicle.channels['2']
print " Ch3: %s" % vehicle.channels['3']
print " Ch4: %s" % vehicle.channels['4']
print " Ch5: %s" % vehicle.channels['5']
print " Ch6: %s" % vehicle.channels['6']
print " Ch7: %s" % vehicle.channels['7']
print " Ch8: %s" % vehicle.channels['8']
print "Number of channels: %s" % len(vehicle.channels)
time.sleep(5)
print "Try to arm!"
vehicle.armed = True
time.sleep(5)
print " Ch1: %s" % vehicle.channels['1']
print " Ch2: %s" % vehicle.channels['2']
print " Ch3: %s" % vehicle.channels['3']
print " Ch4: %s" % vehicle.channels['4']
print " Ch5: %s" % vehicle.channels['5']
print " Ch6: %s" % vehicle.channels['6']
print " Ch7: %s" % vehicle.channels['7']
print " Ch8: %s" % vehicle.channels['8']
print "Number of channels: %s" % len(vehicle.channels)

#start cover
print "\nChannel overrides: %s" % vehicle.channels.overrides
time.sleep(5)
print "Try to change!"
# set1-8 as 110 -810
print "Set Ch1-Ch8 overrides to 110-810 respectively"
vehicle.channels.overrides = {'1':800, '2':800,'3': 800,'4':800, '5':810,'6':610,'7':710,'8':810}
print " Channel overrides: %s" % vehicle.channels.overrides 

time.sleep(5)
print "Try to clear!"
# clear
print "Clear all overrides"
vehicle.channels.overrides = {'1': 0, '2': 0,'3': 0,'4':0, '5':0,'6':0,'7':0,'8':0}
print " Channel overrides: %s" % vehicle.channels.overrides 
time.sleep(5)
print "try to disarm"
vehicle.armed = False

# close vechile
print "\nClose vehicle object"
vehicle.close()
#sitl.stop()
print("Completed")
