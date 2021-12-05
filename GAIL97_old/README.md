# GAIL with Carla 0.9.7

## Requirement

```bash
python==3.5
carla==0.9.7

tensorflow-gpu==1.10.0
tensorboard==1.10.0
tensorboardX==1.8
moviepy==1.0.1 # you'd better first "pip install --update pip"
```

If you are using `conda` to install `tensorflow-gpu`, the corresponding `cudatoolkits`, `cudnn` will be installed autonomously.

If you lack other python packages, please install them manually.

## Usage

- First, open  `carla` using this command:

```bash
./CarlaUE4.sh -carla-port=2000 -quiality-level=Low
```

- Before you start training, please use `auto_pilot_data_collector` to generate expert data, which would take you 2 hours or so.

- You can either run a BC model or a GAIL model.

  - If you want to train BC:

    ```bash
    python main.py --task BC --bc_iters 300 --scene 0 --pose 0 ---weather ClearNoon
    ```

  - If you want to train GAIL

    ```bash
    python main.py --task GAIL --rl_iters 200 --scene 0 --pose 0 --weather ClearNoon
    ```

  - If you want to train RL (disable discriminator)

    ```bash
    python main.py --task RL --rl_iters 200 --discriminator_update 0 --scene 0 --pose 0 --weather ClearNoon
    ```

  - Usually, we train GAIL or RL model based on a pretrained model using BC, like this

    ```bash
    python main.py --task GAIL --rl_iters 500 --scene 0 --pose 0 --weather ClearNoon --load_model BC_scene0_pose0_branch1_ClearNoon_0314_2342_9999
    ```

- You can use `tensorboard` to show the loss, performance, and video

  ```bash
  cd ./log
  tensorboard --logdir .
  ```

## TO-DO List

- batch normalization, advantage normalization
- Modify `reward, done` (change lane, navigation branch, traffic light etc.) in the `Env` to reach a high performance 

## Progress

- new input
- Add `discriminator` to complete GAIL, BC initializer
- change the main framework, now the experiment settings are specified in `config.py` 
- support data collector
- support `tensorboard`, logger, continue training
- support DDPG (code based on `github: HubFire/Multi-branch-DDPG-CARLA`)
  - fix bugs in the actor/critic network
  - tail RGB image, and pass it through CNN
- support Carla 0.9.7 reinforcement learning environment