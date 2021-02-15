# Initial code kept for archive
###############################


# #!/usr/bin/env python3

# import socket
# import time
# import math

# from signal import signal, SIGINT
# from sys import exit

# # HOST = '192.168.1.11'
# HOST = '127.0.0.1'
# PORT = 25001

# def handler(signal_received, frame):
#     global s
#     # Handle any cleanup here
#     print('SIGINT or CTRL-C detected. Exiting gracefully')
#     send_and_receive(s, "stop")
#     s.close()
#     exit(0)

# signal(SIGINT, handler)

# # protocol
# # for each topic/value pair: "key1=value1;key2=value2;key3=value3;" 

# data = ""
# clock = 0
# received = ""
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# def publish(topic, value):
#     return topic+"={};".format(" ".join(map(str,value)))

# def send_and_receive(s, data):
#     global clock, received
#     s.sendall(data.encode("utf-8"))
#     print("SND<<"+data)
#     fragments = []
#     while True:
#         chunk = s.recv(4096).decode("utf-8")
#         fragments.append(chunk)
#         if chunk[-1:] == "@":
#             break
#     received = ''.join(fragments)
#     print("RCV>>"+received[:100])

# def main():
#     global s, data, clock, received

#     time_step = 0.05 # 20hz
#     sleep_time = 0.2
    
#     try:
#         s.connect((HOST, PORT))

#         for _ in range(0,250):
#             # time_in = time.time()
#             data = publish("clock", (clock,) ) + publish("vel_cmd", (1, round(math.cos(clock)*45,3)))
#             send_and_receive(s, data)
#             # time_out = time.time()
#             clock = round( clock + time_step, 3)
#             time.sleep(sleep_time)
#             # if time_out < time_in + time_step :
#             #     time.sleep(time_in + time_step - time_out)

#         clock = 0
#         data = publish("clock", (clock,) ) + publish("sim_control", ('n',))
#         send_and_receive(s, data)

#         for i in range(0,250):
#             # time_in = time.time() 
#             data = publish("clock", (clock,) ) + publish("vel_cmd", (1, math.cos(clock)*45))
#             send_and_receive(s, data)
#             # time_out = time.time()
#             clock = round( clock + time_step, 3)
#             time.sleep(sleep_time)
#             # if time_out < time_in + time_step :
#             #     time.sleep(time_in + time_step - time_out)

#         data = "stop"
#         send_and_receive(s, data)

#     finally:    
#         s.close()

# def data_management(incoming_data, clock):
#     outgoing_data = publish("Clock", clock)
#     return outgoing_data


# if __name__ == '__main__':
#     main()
