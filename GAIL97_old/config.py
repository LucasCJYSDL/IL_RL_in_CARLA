branch_list = [-1, 0, 1]


def get_scene_config(scene_id, pose_id, branch, weather):
    weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'HardRainNoon']

    assert weather in weathers, 'weather should be in set {}'.format(weathers) 
    assert branch in branch_list, 'branch shoule be in set {}'.format(branch_list)

    if scene_id == 0:
        scene = {
            'ped_center' : 741, # waypoint ID
            'ped_range' : 30.0,
            'range' : [-102.0, -60.0, 115.0, 152.0],

            'NumOfWal' : [16, 30], #16 30
            'NumOfVeh' : [0, 0],
            'Wait_ticks' : 10, # TODO: 200

            'start' : [1760, 1917, 1883][pose_id],  # waypoint ID
            'task_type' : [[-1, 1, 0, 2], [-1, 1, 0, 2], [1, -1, 2, 0]][pose_id], # left -1, straight 0, right 1
            'lane_type' : [[0,1,1],[1,1,0],[0,1,1]][pose_id],
            'weather' : weather,
            'branch' : branch,
        }
    else:
        assert False, 'unavailable scene ID'

    return scene


if __name__ == '__main__':
    print(get_scene_config(0, 0, 1, 'ClearNoon'))
