#!/usr/bin/env python3

import socket

def init_socket(HOST,PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    return s

def send_and_receive(s, data, end_char = "@"):
    s.sendall(data.encode("utf-8"))

    # print("Sent")

    fragments = []
    while True:
        chunk = s.recv(4096).decode("utf-8")
        fragments.append(chunk)
        if chunk[-1:] == end_char:
            break
    # print("Received")
    return ''.join(fragments)

def stop(s):
    send_and_receive(s, "stop")
    s.close()