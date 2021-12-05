import h5py

# f = h5py.File('test.h5', 'w')
#
# g = f.create_group('scene')
# h = g.create_group('pose')
# p = h.create_group('branch')


file_name = './dataset/intersection_0505_0403.h5'
with h5py.File(file_name, 'r') as f:
    # for key in f.keys():
    #     # print(f[key])
    #     for sub_key in f[key].keys():
    #         # print(f[key][sub_key])
    #         for sub_sub_key in f[key][sub_key].keys():
    #             # print(f[key][sub_key][sub_sub_key])
    #             for sss_key in f[key][sub_key][sub_sub_key].keys():
    #                 print(f[key][sub_key][sub_sub_key][sss_key], type(f[key][sub_key][sub_sub_key][sss_key]))


    for key in f['scene_0/pose_0/branch_0/episode_0'].keys():
        if key=='evaluation_metrics':
            continue
        print(f['scene_0/pose_0/branch_0/episode_0'][key])

    for key in f['scene_0/pose_0/branch_0/episode_0/evaluation_metrics']:
        print(f['scene_0/pose_0/branch_0/episode_0/evaluation_metrics'][key], f['scene_0/pose_0/branch_0/episode_0/evaluation_metrics'][key][:])
