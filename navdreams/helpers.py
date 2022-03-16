import numpy as np
import traceback

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

def get_goal(dico):
    if "goal" in dico:
        try:
            goal = np.array(list(map(float,dico["goal"].split(' '))))
            return goal
        except ValueError:
            print(dico["goal"])
            traceback.print_exc()
            return None
    return None

def get_walls(dico):
    walls = np.array([])
    if "walls" in dico:
        try:
            _,vertxy = dico['walls'].split(' ', 1)
            walls = np.array(list(map(float,vertxy.split(' ')))).reshape((-1, 4, 2))
        except ValueError:
            print(dico["walls"])
            traceback.print_exc()
            return np.array([])
    return walls

def get_trialinfo(dico):
    if "trial" in dico:
        try:
            name = dico["trial"]
            return name
        except: # noqa
            print(dico["trial"])
            traceback.print_exc()
            return None
    return None

def get_crowd(dico):
    ITEMS = ["id", "x", "y", "theta"]
    N_ITEMS = len(ITEMS)
    THETA_IDX = ITEMS.index("theta")
    if "crowd" in dico:
        try:
            if dico['crowd'] == "id pose":
                return None
            _,_,crowd = dico['crowd'].split(' ', 2) # removes "id pose" at beginning
            crowd = crowd.replace('(', '')
            crowd = crowd.replace(')', '')
            crowd = np.array(list(map(float,crowd.split(' ')))).reshape((-1,N_ITEMS))
            # unity agent x axis is lateral right -> -90 degrees to get anterior axis
            # unity rotation is in degrees -> deg2rad
            # unity rotation is counter-trigonometric-angle -> minus sign
            crowd[:, THETA_IDX] = -np.deg2rad(crowd[:, THETA_IDX]-90)
            return crowd
        except ValueError:
            print(dico['crowd'])
            traceback.print_exc()
            return None
    return None

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
            traceback.print_exc()
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
        except ValueError as ve: # noqa
            pass
        except IndexError as ie: # noqa
            pass

    return False

def sim_control(input):
    return {"clock" : 0 , "sim_control" : input}

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
