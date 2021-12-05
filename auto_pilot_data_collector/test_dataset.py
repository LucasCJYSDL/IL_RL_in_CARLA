import h5py

file_name = 'ExpertData/scene_0.h5'
with h5py.File(file_name, 'r') as f:

    for key in f.keys():
        print(f[key])
        for sub_key in f[key].keys():
            print(f[key][sub_key])
