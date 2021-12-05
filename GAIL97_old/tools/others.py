import numpy as np
import cv2
import os

def OU(x, mu, sigma, theta):
    """
    x, mu, sigma, theta
    """
    return theta * (mu - x) + sigma * np.random.randn(1)

def render_image(name, img):
    """using cv2
    
    Arguments:
        name {str}
        img {np.array}
    """
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(1)


def print_dict(name, dic):
    print(name)
    for (k, v) in dic.items():
        print('\t', k, ' : ', v)


'''
def save_variables(save_path, variables, sess):

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)

def load_variables(load_path, variables, sess):

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))
    sess.run(restores)
'''