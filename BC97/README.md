# BC with Carla 0.9.7

### Requirement

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

### Open Carla

whether you want to collect data, train models, or evaluate performance, the first thing that hopefully you  don't forget is to open  `carla` using this command:

```bash
./CarlaUE4.sh -carla-port=2000 -quiality-level=Low
```

Or you can run off-screen for lower GPU utility

```bash
DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000 -quiality-level=Low
```

### Collect data

You can run `data_collector.py` to collect trajectories which are generated by our `PID` controller.

The collected data would be stored in a replay buffer (see `tools/buffer.py` for more help), and this data can be used for both behavior cloning and off-policy reinforcement learning algorithms.

You can specify `scenes_ID`,  `iters`,  and the controllers of `lateral` and `longitude`, like this:

```bash
python data_collector.py --scenes 0 1 --iters 10 --lateral PID_NOISE --longitude PID
```

**NOTES**:  `PID_NOISE` would add a uniform noise between `[-0.2, 0.2]` to `steer` when collecting data.

The saved data file would be `scene01_XX_XX.5` in the `dataset` file. It would take you several hours to complete collecting.

### Behavior Cloning

You can run `main.py` to train a behavior cloning model . 

**NOTES:** Don't forget to specify `--task` , `--load_data` and `--bc_iters`. 

```bash
python main.py --task bc_test --load_data xxx.h5 --bc_iters 1000
```

**Comments**:

- If you want to continue training a BC model, you can input `--bc_model XXX/XXX_X`, like `BC_test_0411_1212/BC_test_0411_1212_999`.
- You can use `tensorboard` to see the performance and video in the training stage

```bash
tensorboard --logdir ./log
```

### Reinforcement Learning

After you completing training `BC`, you can use it as a base model and continue to train a `RL` model.

In our algorithm, our `actor` network would load the parameters from the `bc_model`, and in the first `rl_pre_iters` episodes, we would fix `actor` and solely train `critic`. Then the agent begins to interact with th environment.

```bash
python main.py --task rl_test --load_data xxx.h5 --bc_model XXX/XXX_X --rl_iters 200 --rl_pre_iters 20 
```

**Comments**:

- `--rl_model`:  You can continue training from a rl model.
- `--rl_scenes`: After `rl_pre_iters` episodes, `actor` would be used to interact with environment and collect data from the given `scenes`.

### Evaluate Performance

After you have trained a model successfully, you can run `evaluate_model.py` to get a more detailed report for your model. And the report would be issued in the `./evaluation` folder.

```bash
python evaluate_model.py --scenes 0 2 --iters 10 --lateral NN --longitude NN --nn_model XXX/XXX_X
```

**Note:** If you can want test `baseline` model, like `PID`, you can specify `--lateral PID` and `--longitude NN`.

### Modify Configuration

Our experimental scenarios are specified in `configer.py`.  If you want to add more scenarios, please follow the `__init__` format of `class scene`.

### To-Do

- support RL module(batch normalization, advantage normalization, GAE)
- Modify `reward, done` (change lane, navigation branch, traffic light etc.) 
- GAIL