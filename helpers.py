#inputs and outputs
import numpy as np

def publish_all(dico):
    to_publish = []
    for key, value in dico.items():
        to_publish.append(publish(key,value))
    return ''.join(to_publish)
        
def publish(topic, value):
    if type(value) is tuple:
        return topic+"={};".format(" ".join(map(str,value)))
    else:
        return topic+"={};".format(value)

def raw_data_to_dict(received):
    dico = {'step_number': 0}

    raw_topics = received.split(';')
    for raw_topic in raw_topics:
        if raw_topic != "@": # end of raw data
            topic, rest = raw_topic.split('=',1)
            step_number, value = rest.split('#',1)

            dico['step_number'] = step_number
            dico[topic] = value

    return dico

def get_crowd(dico):
    crowd = np.array([])
    if "crowd" in dico:
        try:
            _,_,crowd = dico['crowd'].split(' ', 2)
            crowd = crowd.replace('(', '')
            crowd = crowd.replace(')', '')
            crowd = np.array(list(map(float,crowd.split(' ')))).reshape((-1,3))
        except ValueError:
            return np.array([])
    return crowd

def get_clock(dico):
    clock = -1
    if "clock" in dico:
        clock = round(float(dico["clock"]),3)
    return clock

def get_odom(dico):
    if "odom" in dico:
        try:
            odom_xytheta_dxdydtheta = list(map(float, dico['odom'].split(" ")[1:]))
        except ValueError:
            pass
    else:
        odom_xytheta_dxdydtheta = []

    deg_angle = odom_xytheta_dxdydtheta[2]
    deg_rotvel = odom_xytheta_dxdydtheta[5]
    odom_xytheta_dxdydtheta[2] = np.deg2rad(deg_angle)
    odom_xytheta_dxdydtheta[5] = np.deg2rad(deg_rotvel)
    return np.array(odom_xytheta_dxdydtheta)

# simulation controls and checks
def do_step(step, dico):
    if "clock" in dico:
        dico["clock"] = round(dico["clock"] + step,3)
    return dico

def check_ending_conditions(max_time, min_x, dico):
    if "clock" in dico:
        if float(dico["clock"]) < 1:
            return False

    if "clock" in dico:
        if float(dico["clock"]) > max_time:
            return True
        
    if "odom" in dico:
        try:
            _, rest = dico["odom"].split(' ',1)
            odom_x, _ = rest.split(' ',1)
            if float(odom_x) < min_x:
                return True
        except ValueError as ve:
            pass
        except IndexError as ie:
            pass

    return False

def sim_control(input):
    return { "clock" : 0 , "sim_control" : input }

def reset():
    return sim_control('r')

def next():
    return sim_control('n')

def previous():
    return sim_control('p')

def first():
    return sim_control('f')

def last():
    return sim_control('l')

def stop():
    return sim_control('l')

def idle():
    return sim_control('i')
