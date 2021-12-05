import h5py
import os
import numpy as np
from PIL import Image
import carla
import math

def write_hdf5(file_name, group_name, sensors_list, measures_list, task_type_list, lane_type_list, center):

    assert sensors_list.shape[0] == measures_list.shape[0]
    lent = sensors_list.shape[0]
    result = {}
    keys = ['FrontRGB', 'Lidar', 'Measurement', 'control']
    for key in keys:
        result[key] = []
    for i in range(lent):
        result['FrontRGB'].append(sensors_list[i]['FrontRGB'])
        # result['FrontSemantic'].append(sensors_list[i]['FrontSemantic'])
        result['Lidar'].append(sensors_list[i]['Lidar'])
        result['Measurement'].append(get_measurement(i, measures_list, lane_type_list, center))
        result['control'].append([measures_list[i]['control']['steer'], measures_list[i]['control']['throttle'], measures_list[i]['control']['brake']])


    with h5py.File(file_name, 'a') as f:
        group = f.create_group(group_name)
        for key in keys:
            print(key)
            print(np.array(result[key]).shape)
            # print(np.array(result[key]))
            group.create_dataset(key, data=np.array(result[key]), dtype=np.array(result[key]).dtype)
        group.create_dataset('task_type', data=task_type_list)

def get_measurement(ind, measures_list, lane_type_list, center):

    measurement = []
    measurement.extend([measures_list[ind]['speed']])
    print("center: ", center[0], " ", center[1])
    print("vehicle: ", measures_list[ind]['location'].x, " ",measures_list[ind]['location'].y)
    measurement.extend([measures_list[ind]['location'].x-center[0], measures_list[ind]['location'].y-center[1]])
    print("yaw: ", measures_list[ind]['rotation'].yaw)
    angle = np.radians(measures_list[ind]['rotation'].yaw)
    while (angle > math.pi):
        angle -= 2 * math.pi
    while (angle < -math.pi):
        angle += 2 * math.pi
    assert angle>=-np.pi and angle<=np.pi
    measurement.extend([angle])
    measurement.extend(lane_type_list)
    print(measurement)

    return np.array(measurement)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def pre_process(rgb_image):

    rgb_image = rgb_image[115 : 510, :]

    image = np.array(Image.fromarray(rgb_image).resize((200, 88)))

    # image = image.astype(np.float32)
    # image = np.multiply(image, 1.0 / 255.0)
    print(image.shape)

    return image

def draw_waypoint_union(debug, l0, l1, to_show, color=carla.Color(0, 0, 255), lt=100):
    debug.draw_line(l0, l1, thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(l1, 0.1, color, lt, False)
    debug.draw_string(l1, str(to_show), False, carla.Color(255, 162, 0), 200, persistent_lines=False)


def get_matrix(rotation):
    """
    Creates matrix from carla transform.
    """
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    matrix = np.matrix(np.identity(2))
    # matrix[0, 0] = c_y
    # matrix[0, 1] = -s_y
    # matrix[1, 0] = s_y
    # matrix[1, 1] = c_y
    matrix[0, 0] = -s_y
    matrix[0, 1] = -c_y
    matrix[1, 0] = c_y
    matrix[1, 1] = -s_y
    return matrix

def sensor_to_world(cords, sensor_transform):
    """
    Transforms world coordinates to sensor.
    """
    sensor_world_matrix = get_matrix(sensor_transform.rotation)
    print(sensor_world_matrix)
    world_cords = np.dot(sensor_world_matrix, np.transpose(cords))
    world_cords[0,0] += sensor_transform.location.x
    world_cords[0,1] += sensor_transform.location.y
    world_cords = world_cords.tolist()
    return world_cords[0]

def get_distance(point):
    d_x = point[0]
    d_y = point[1]
    dis = math.sqrt(d_x*d_x+d_y*d_y)
    if dis < 0.1:
        print("Too close! Impossible!!", d_x, " ", d_y)
        return 1.0
    rel_dis = dis/20.0
    print("rel_dis", rel_dis)
    rel_dis = min(1.0, rel_dis)

    return rel_dis

def get_angle(point):

    if point[0] - 0.0 < 1e-3 and point[0] > 0.0:
        point[0] = 1e-3
    if 0.0 - point[0] < 1e-3 and point[0] < 0.0:
        point[0] = -1e-3

    angle = math.atan2(point[1], point[0])
    while (angle > math.pi):
        angle -= 2 * math.pi
    while (angle < -math.pi):
        angle += 2 * math.pi
    assert angle>=-np.pi and angle<=np.pi
    degree = math.degrees(angle)
    print("degree", degree)

    rel_degree = int((degree+360.0) % 360)
    print("rel_degree", rel_degree)

    return rel_degree