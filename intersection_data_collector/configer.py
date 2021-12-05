"""
Configer is used to return the environment setting
"""


branch_list = [-1, 0, 1]
weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'HardRainNoon']

def Configer(scene_id):
    if scene_id == 0:
        return scene0()
    elif scene_id == 1:
        return scene1()
    elif scene_id == 2:
        return scene2()
    else:
        assert False



#############################
#   Scene Configuration
############################

class scene:

    def __init__(self, ped_center, ped_range, area, NumOfWal, NumOfVeh, poses, ends, task_type, lane_type):
        self.ped_center = ped_center
        self.ped_range = ped_range
        self.area = area

        self.NumOfWal = NumOfWal
        self.NumOfVeh = NumOfVeh

        self.poses = poses
        self.ends = ends
        self.task_type = task_type
        self.lane_type = lane_type

    def poses_num(self):
        return len(self.poses)
    
    def branches(self, pose):
        assert pose in set(range(self.poses_num()))

        branches = []
        for b, v in zip(branch_list, self.lane_type[pose]):
            if v == 1:
                branches.append(b)
        
        return branches

    def scene_config(self, pose, branch):
        assert pose in set(range(self.poses_num()))
        assert branch in self.branches(pose), "not consistent with lane type"

        scene = {
            'ped_center' : self.ped_center,
            'ped_range' : self.ped_range,
            'area' : self.area,

            'Wait_ticks' : 50,
            'weather' : 'ClearNoon',

            'NumOfWal' : self.NumOfWal,
            'NumOfVeh' : self.NumOfVeh,

            'start' : self.poses[pose], 
            'end' : self.ends[pose][branch_list.index(branch)],
            'task_type' : self.task_type[pose],
            'lane_type' : self.lane_type[pose],
            'branch' : branch,
        }
        return scene
    



class scene0(scene):

    def __init__(self):
        super().__init__(
            ped_center = 741,
            ped_range = 30.0,
            area = [-102.0, -60.0, 115.0, 152.0],

            NumOfWal = [20, 30],
            NumOfVeh = [0, 0],

            poses = [1760, 1917, 1883],
            ends = [[-1, 1844, 1367], [2327, 2001, -1], [-1, 544, 2327]],
            task_type = [[-1, 1, 0, 2], [-1, 1, 0, 2], [1, -1, 2, 0]],
            lane_type = [[0, 1, 1], [1, 1, 0], [0, 1, 1]],
        )



class scene1(scene):

    def __init__(self):
        super().__init__(
            ped_center = 610,
            ped_range = 30.0,
            area = [-27.0, 25.0, 111.0, 153.0],

            NumOfWal = [20, 30],
            NumOfVeh = [0, 0],

            poses = [1219, 593, 2015],
            ends = [[-1, 2719, 1234], [1997, 79, -1], [1234, 52, -1]], 
            task_type = [[-1, 1, 0, 2], [-1, 1, 0, 2], [1, -1, 2, 0]],
            lane_type = [[0, 1, 1], [1, 1, 0], [1, 1, 0]],
        )


class scene2(scene):

    def __init__(self):
        super().__init__(
            ped_center = 1374,
            ped_range = 30.0,
            area = [-18.0, 25.0, -155.0, -117.0],

            NumOfWal = [20, 30],
            NumOfVeh = [0, 0],

            poses = [2737, 1927, 1725, 2204],
            ends = [[-1, 801, 353], [95, 1307, -1], [353, 346, -1], [-1, 847, 95]], 
            task_type = [[-1, 1, 0, 2], [-1, 1, 0, 2], [1, -1, 2, 0], [1, -1, 2, 0]],
            lane_type = [[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]],
        )