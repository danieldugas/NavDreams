#!/usr/bin/env python3

import socket
import select

def init_socket(HOST,PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    return s

def send_and_receive(s, data, end_char = "@"):
    timeout_s = 1

    s.sendall(data.encode("utf-8"))

    # print("Sent")

    fragments = []
    while True:
        # add timeout
        s.setblocking(0)
        ready = select.select([s], [], [], timeout_s)
        if ready[0]:
            chunk = s.recv(4096).decode("utf-8")
        else:
            print("recv timed out. retrying")
            continue
        fragments.append(chunk)
        if chunk[-1:] == end_char:
            break
    # print("Received")
    raw = ''.join(fragments)
    return raw

def stop(s):
    send_and_receive(s, "stop")
    s.close()