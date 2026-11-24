"""
Simplest laser test: send one byte over UDP to the Raspberry Pi running laser_udp_test.slx.
  1 = laser on, 0 = laser off.

The laser turns on, stays on until you press Enter, then turns off.
"""

import socket

PI_IP = "138.38.226.64"   # Pi's WiFi IP (on the Pi: hostname -I) - update if it changes
PI_PORT = 50001           # must match the UDP Receive block's port in laser_udp_test.slx

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.sendto(bytes([1]), (PI_IP, PI_PORT))
print(f"Sent 1 to {PI_IP}:{PI_PORT} -> laser should be ON")

input("Press Enter to turn the laser off...")

# The UDP Receive block keeps its last value, so we must explicitly send 0 to turn it off
sock.sendto(bytes([0]), (PI_IP, PI_PORT))
print("Sent 0 -> laser should be OFF")
sock.close()
