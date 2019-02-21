# Introduction
It is the github repo for the paper: [NerveNet: Learning Structured Policy with Graph Neural Networks](http://www.cs.toronto.edu/~tingwuwang/nervenet.html).
# Dependency

The repo is written in Python 2.7. You might need to modify the code repo for compatibility in Python 3.x. Sorry for the inconvenience!

## 1. tensorflow >= 1.0.1
```bash
pip install tensorflow-gpu
```
GPU version is not mandatory, since in the current repo, gpu is not used by default.
## 2. gym >= 0.7.4
### gym dependency
```bash
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

### gym installation via pip
```bash
pip install 'gym[mujoco]'
```
To use the mujoco, we actually need to use the mjkey.txt
## 3. mujoco
```bash
pip install mujoco-py==0.5.7
```
Note that currently, we **only** support **MJPro 1.31**.
Please install mujoco 1.31 from the [official website](http://www.mujoco.org/), and use the mujoco-py version **0.5.7**.
## 4. Misc
```bash
pip six beautifulsoup4 termcolor num2words
```
# Run the code
To run the code, first cd into the 'tool' directory.
We provide three examples below (The checkpoint files are already included in the repo):

To test the transfer learning result of **MLPAA** from *centipedeSix* to *centipedeEight*:
```bash
python main.py --task CentipedeEight-v1 --use_gnn_as_policy 0 --num_threads 4 --ckpt_name ../checkpoint/centipede/fc/6 --mlp_raw_transfer 1 --transfer_env CentipedeSix2CentipedeEight  --test 100
```
You should get the average reward around *20*. If you want to test the performance of pretrained models, you should use:
```bash
python main.py --task CentipedeSix-v1 --use_gnn_as_policy 0 --num_threads 4 --ckpt_name ../checkpoint/centipede/fc/6 --mlp_raw_transfer 1  --test 100
```
The performance of the pretrained model of **MLPAA** is around *2755*.

Similarly, to get the transfer learning result of **NerveNet** from *centipedeSix* to *centipedeEight*:
```bash
python main.py --task CentipedeEight-v1 --use_gnn_as_policy 1 --num_threads 4 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB --ckpt_name ../checkpoint/centipede/gnn/6 --transfer_env CentipedeSix2CentipedeEight --test 100
```
The reward of **NerveNet** should be around *1600*. And to test the pretrained model:
```bash
python main.py --task CentipedeSix-v1 --use_gnn_as_policy 1 --num_threads 4 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB --ckpt_name ../checkpoint/centipede/gnn/6 --test 100
```
The reward for **NerveNet** pretrained model is around: *2477*

To train an agent from sratch using NerveNet, you could use the following code:
```bash
python main.py --task ReacherOne-v1 --use_gnn_as_policy 1 --network_shape 64,64 --lr 0.0003 --num_threads 4 --lr_schedule adaptive --max_timesteps 1000000 --use_gnn_as_value 0 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB
```
