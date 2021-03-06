# GAIL (Carla 0.9.7)

### Requirement

```bash
python==3.5
carla==0.9.7

tensorflow-gpu==1.10.0
tensorboard==1.10.0
tensorboardX==1.8
moviepy==1.0.1	# you'd better first "pip install --update pip"
```

If you are using `conda` to install `tensorflow-gpu`, the corresponding `cudatoolkits`, `cudnn` will be installed autonomously.

### Open Carla

You can use this command to open  `carla` with or without display.

```bash
./CarlaUE4.sh -carla-port=2000 -quiality-level=Low						# with display

DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000 -quiality-level=Low		# without display
```

### Collect data

You can run `data_collector.py` to collect trajectories which are generated by our `PID` controller. The dataset will be stored as `.h5` files. 

```bash
python data_collector.py --scenes 0 1 --iters 10 --lateral PID_NOISE --longitude PID
```

**NOTES**:  `PID_NOISE` would add a uniform noise between `[-0.2, 0.2]` to `steer` when collecting data.

### Convert Raw data

If you are using the data collected manually, you can run `convert_raw_data.py` .

```bash
python convert_raw_data.py
```

### Behavior Cloning

You can run `main.py` to train a behavior cloning model . 

```bash
python main.py --task bc_test --load_data xxx.h5 --bc_iters 1000
```

**NOTES**:

- If you want to continue training a BC model, you can input `--bc_model XXX/XXX_X`, like `BC_test_0411_1212/BC_test_0411_1212_999`.
- You can use `tensorboard` to see the performance and video during training.

```bash
tensorboard --logdir .
```

### Reinforcement Learning

After you completing training `BC`, you can use it as a pretrained model and continue to train a `RL` model.

```bash
python main.py --task rl_test --load_data xxx.h5 --bc_model XXX/XXX_XX --rl_iters 100 --replay 4000 
```

**NOTES**:

- `--rl_model`:  You can continue training from a rl model.

### GAIL

```bash
python main.py --task gail_test --load_data xxx.h5 --bc_model xxx --gail_iters 100 --replay 4000
```

### Evaluate Performance

After you have trained a model successfully, you can run `evaluate_model.py` to get a more detailed report for your model. And the report would be issued in the `./evaluation` folder.

```bash
python evaluate_model.py --scenes 0 2 --iters 10 --lateral NN --longitude NN --nn_model XXX/XXX_X
```

**Note:** If you can want test `baseline` model, like `PID`, you can specify `--lateral PID` and `--longitude NN`.

### Modify Configuration

Our experimental scenarios are specified in `configer.py`.  If you want to add more scenarios, please follow the `__init__` format of `class scene`.
