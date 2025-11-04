# drl_th
## Setup

The implementation of the entire project depends on the following environments:

- **ROS Noetic**
- **carla 0.9.12**
- **carla-ros-bridge 0.9.12**

Please ensure that Carla and the ROS bridge are successfully configured on your system before proceeding with the subsequent environment configurations. The following content may help you complete the setup of the above environments.

We are using the **carla-0.9.12 Released version**.  
Download [carla-0.9.12 and AdditionalMaps-0.9.12](https://github.com/carla-simulator/carla/releases).  
Then you can follow the steps below to configure the simulation environment.

---

## 1 Carla Config

```bash
mkdir carla12
cd carla12
tar -xf CARLA_0.9.12.tar.gz
tar -xf AdditionalMaps_0.9.12.tar.gz
```

Then, you can configure your environment variables by following these steps.

```bash
# Input these lines into your ~/.bashrc
export CARLA_ROOT="(where your carla)/carla12"
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/examples
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI

# we write two alias to open Carla in two modes [render on or off]
alias carla="(where your carla)/carla12/CarlaUE4.sh"
alias carla_no_rendering="(where your carla)/carla12/CarlaUE4.sh -RenderOffScreen"
```

## 2 Environment

### 2.1 launch carla
```bash
carla_no_rendering
```
### 2.2 launch carla-ros-bridge
```bash
conda activate drl     # your conda virtual environment
roslaunch carla_ros_bridge drl_th_town02.launch   # mainly external sensor configuration, etc.
```
### 2.3 train and test
```bash
python train.py
python test.py
```

The above is the baseline, other details are being added...
