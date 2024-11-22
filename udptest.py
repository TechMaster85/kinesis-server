import sys
import socket
 
UDP_IP = "35.3.236.152"
UDP_PORT = 35155
MESSAGE = b"Hello, World!"

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
