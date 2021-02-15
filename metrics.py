#!/usr/bin/env python3
import recorder
import numpy as np
import matplotlib.pyplot as plt


#TODO comments!!!

def eucl_dist(a,b):
    return np.linalg.norm(np.array(b)-np.array(a))

def compute_path_efficiency(alone_json, crowded_json):
    #T alone / T crowded
    #L alone / L crowded
    #J alone / J crowded

    print(crowded_json)
    alone = recorder.load_file_as_list_of_dico(alone_json)
    crowded = recorder.load_file_as_list_of_dico(crowded_json)
    
    Talone = float(alone[0]['clock']) - float(alone[-1]['clock'])
    Tcrowded = float(crowded[0]['clock']) - float(crowded[-1]['clock'])

    alone_odom_xy = [ list(map(float, alone[i]['odom'].split(" ")[1:3])) for i,_ in enumerate(alone) ]
    crowded_odom_xy = [ list(map(float, crowded[i]['odom'].split(" ")[1:3])) for i,_ in enumerate(crowded) ]

    Lalone = sum( [ eucl_dist(alone_odom_xy[i], alone_odom_xy[i+1]) for i in range(len(alone_odom_xy)-1) ] )
    Lcrowded = sum( [ eucl_dist(crowded_odom_xy[i], crowded_odom_xy[i+1]) for i in range(len(crowded_odom_xy)-1) ] )


    alone_odom = [ list(map(float, alone[i]['odom'].split(" ")[1:4])) for i,_ in enumerate(alone) ] 
    crowded_odom = [ list(map(float, crowded[i]['odom'].split(" ")[1:4])) for i,_ in enumerate(crowded) ] 

    Jalone = np.sum(np.abs(np.diff(np.diff(np.reshape(np.ravel(alone_odom, order='F'),(3,-1))))))
    Jcrowded = np.sum(np.abs(np.diff(np.diff(np.reshape(np.ravel(crowded_odom, order='F'),(3,-1))))))
    
    return (Talone/Tcrowded , Lalone/Lcrowded, Jalone/Jcrowded)

def compute_effect_on_crowd(not_reactive_crowd_json, reactive_crowd_json):
    not_reactive_crowd = recorder.load_file_as_list_of_dico(not_reactive_json)
    reactive_crowd = recorder.load_file_as_list_of_dico(reactive_json)

    # Tcrowded = ??
    # Tcrowd_alone = ??

    data = []
    for i in range(len(not_reactive_crowd)):
        _,_,crowd = not_reactive_crowd[i]['crowd'].split(' ', 2)
        crowd = crowd.replace('(', '')
        crowd = crowd.replace(')', '')
        crowd = np.array(map(float,crowd.split(' '))).reshape((-1,3))
        data.append(crowd)
    data = np.array(data)

    not_reactive_crowd_traj = np.array([  [ data[i][j][1:] for i in range(len(data)) ] for j in range(len(data[0])) ])

    not_reactive_crowd_dist = np.array([ [ eucl_dist(not_reactive_crowd_traj[i][j], not_reactive_crowd_traj[i][j+1]) for j in range(len(not_reactive_crowd_traj[i])-1)] for i in range(len(not_reactive_crowd_traj))])
    not_reactive_crowd_vel = np.diff( np.where( np.abs(not_reactive_crowd_dist) < 5, not_reactive_crowd_dist, 0))

    Vel_not_reactive_crowd = np.mean(not_reactive_crowd_vel)

    data = []
    for i in range(len(reactive_crowd)):
        _,_,crowd = reactive_crowd[i]['crowd'].split(' ', 2)
        crowd = crowd.replace('(', '')
        crowd = crowd.replace(')', '')
        crowd = np.array(map(float,crowd.split(' '))).reshape((-1,3))
        data.append(crowd)
    data = np.array(data)

    reactive_crowd_traj = np.array([  [ data[i][j][1:] for i in range(len(data)) ] for j in range(len(data[0])) ])
    reactive_crowd_dist = np.array([ [ eucl_dist(reactive_crowd_traj[i][j], reactive_crowd_traj[i][j+1]) for j in range(len(reactive_crowd_traj[i])-1)] for i in range(len(reactive_crowd_traj))])
    reactive_crowd_vel = np.diff( np.where( np.abs(reactive_crowd_dist) < 5, reactive_crowd_dist, 0))

    Vel_reactive_crowd = np.mean(reactive_crowd_vel)


    #Neighbors

    neighbors_distance = 2
    Vel_neighbors = np.array([])

    reactive_odom_xy = [ map(float, reactive_crowd[i]['odom'].split(" ")[1:3]) for i,_ in enumerate(reactive_crowd) ] 

    is_near = np.array([[eucl_dist(reactive_odom_xy[j], reactive_crowd_traj[i][j]) < 2 for j in range(len(reactive_crowd_traj[i])-1)] for i in range(len(reactive_crowd_traj))])

    neighbor_vel = np.array([])
    for top_index,top_val in enumerate(reactive_crowd_vel):
        for index,val in enumerate(top_val):
            if is_near[top_index][index]:
                neighbor_vel = np.append(neighbor_vel, val )
    
    Vel_reactive_neighbors = np.mean(neighbor_vel)

    return (Vel_not_reactive_crowd/Vel_reactive_crowd, Vel_reactive_neighbors/Vel_reactive_crowd)



def compute_crowd_robot_interaction(scenario_json):
    #proximity 1
    scenario = recorder.load_file_as_list_of_dico(scenario_json)

    odom_xy = [ map(float, scenario[i]['odom'].split(" ")[1:3]) for i,_ in enumerate(scenario) ] 

    data = []
    for i in range(len(not_reactive_crowd)):
        _,_,crowd = not_reactive_crowd[i]['crowd'].split(' ', 2)
        crowd = crowd.replace('(', '')
        crowd = crowd.replace(')', '')
        crowd = np.array(map(float,crowd.split(' '))).reshape((-1,3))
        data.append(crowd)
    data = np.array(data)

    crowd_id_xy = np.array([  [ data[i][j][1:] for i in range(len(data)) ] for j in range(len(data[0])) ])

    relative_distances = np.array([[eucl_dist(odom_xy[j], crowd_id_xy[i][j]) for j in range(len(reactive_crowd_traj[i])-1)] for i in range(len(reactive_crowd_traj))])

    Proxity = np.array([])

    for step in range(len(relative_distances[0])): #time values
        values_at_this_step = np.array([])
        for id_index, time_values in enumerate(relative_distances):
            values_at_this_step = np.append(values_at_this_step, time_values[step])
        
        Proximity = np.append(Proximity, 1./np.amin( values_at_this_step ))

    return np.sum(Proxity)

def make_radar_chart(name, stats, attribute_labels, plot_markers, plot_str_markers):

    labels = np.array(attribute_labels)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))

    fig= plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    plt.yticks(plot_markers)
    ax.set_title(name)
    ax.grid(True)

    fig.savefig("images/%s.png" % name)

    return plt.show()
