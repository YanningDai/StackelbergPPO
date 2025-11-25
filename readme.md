# Efficient Morphology–Control Co-Design via Stackelberg PPO

Official implementation of the paper **“Efficient Morphology–Control Co-Design via Stackelberg PPO under Non-Differentiable Leader–Follower Interfaces”**.

<img src="static/m.png" alt="description">

## 📦 Installation

**System Requirements**

- Tested OS: Linux Ubuntu 24.04.3 LTS
- Python >= 3.9
- PyTorch == 2.0.1

**Dependencies**

1. Clone this GitHub repository and enter the project directory:

```bash
cd StackelbergPPO
```
2. Create the Conda environment and install dependencies
```bash
conda create -n StackelPPO python=3.9 -y
conda activate StackelPPO
conda install mesalib glew glfw patchelf -c conda-forge -y
pip install -r requirements.txt
```

3. Install MuJoCo 2.1.0

```bash
mkdir -p ~/.mujoco
cd /tmp
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

4. Add Environment Variables

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
```

## 🚀 Quick Start

**Single environment training**

To train a single environment:
```bash
OMP_NUM_THREADS=1 python -m design_opt.train cfg=pusher
```
**Available environments**: cheetah, crawler, glider-hard, glider-medium, glider-regular, pusher, stepper-hard, stepper, swimmer, terraincrosser, walker-hard, walker-medium, walker-regular

Customize the output directory:
```bash
OMP_NUM_THREADS=1 python -m design_opt.train cfg=pusher hydra.run.dir="single_run/pusher"
```

## ⚙️ Advanced Usage

**Resume training from checkpoint**

Continue training from a previous checkpoint:

```bash
OMP_NUM_THREADS=1 python -m design_opt.train cfg=pusher +restore_dir="single_run/test"
```
Load only the morphology prior without controller weights:
```bash
OMP_NUM_THREADS=1 python -m design_opt.train cfg=pusher +restore_dir="single_run/test" morph_prior=true
```
**Visualization and evaluation**

Visualize learned policies from a checkpoint:
```bash
python design_opt/eval.py --restore_dir single_run/pusher
```

**Configuration**

This project uses Hydra for configuration management. Key configuration files are located in the design_opt/conf/config.yaml directory. Modify configurations either through YAML files or command-line overrides:

```bash
python -m design_opt.train cfg=pusher lamda=5 gradient_ratio_limit=1.0
```

## 📊 Visualization and Results

All experimental visualizations are hosted at this anonymous project page: <https://regalmoxie034f46-anonymous.netlify.app>

Visit the website to explore:
- Evolved morphologies and visualization across diverse robot locomotion and manipulation tasks
- Performance metrics and comparisons with state-of-the-art methods
- Additional experiments on extended environments

## 🙏 Acknowledgements

This project builds upon and is inspired by [BodyGen](https://github.com/Josh00-Lu/BodyGen) and [Transform2Act](https://github.com/Khrylx/Transform2Act). We gratefully acknowledge their excellent work! 

