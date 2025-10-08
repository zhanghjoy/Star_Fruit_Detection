# 🚀 A Lightweight Star Fruit Quality Detector Based on a Label-Balanced Strategy and Deployed on Real-Time Edge Devices 🚀

👨‍💻 Author: [Haijun Zhang](https://github.com/zhanghjoy)  
🏛️ Affiliation: Guangxi University
------

## 📌 Updates Record

- ✅ **2025/07/11** — Build this repo to record core file on my paper.
- ✅ **2025/09/02** — Updated utility files: `Star_Fruit_Detection/src/utils`
- ✅ **2025/09/16** — Updated modified modules and network YAML configuration files.
- ✅ **2025/09/17** — Updated core pruning EXP code ：`Star_Fruit_Detection/src/core/prune.py`;Upload requirements.txt ;Create Repo License
- ✅ **2025/10/07** — The uploaded source code is based on a PyQt5-developed UI interface, which supports selecting both .pt and .engine format weight files, as well as recognition in video, image, and camera modes. A preview of the UI interface is shown in the figure below.
<img width="1202" height="839" alt="48ab5c62d6fb6b8c5b55cde73d19fe30" src="https://github.com/user-attachments/assets/fd65f053-2040-4a7c-958f-4e9a1ec3467b" />

------

## 📖 Abstract

This repository contains the code, tools, and reference materials related to the paper.

------

## 🔰 0. Preface

- **src**: Contains utility scripts, label-balancing code, segmentation tools, image augmentation scripts, label statistics, improved modules, and configuration files.
- **photo**: Stores processed dataset images.#TODO🤷‍♂️
- **docs**: Stores some doc file.
- **Open dataset**: [Mendeley Data](https://data.mendeley.com/datasets/f35jp46gms/1)
- **Annotation tool**: [labelImg](https://github.com/tzutalin/labelImg)

------

## ⚙️ 1. Environment Setup

The training environment used in this work:

| Component   | Version/Spec                |
| ----------- | --------------------------- |
| OS          | Windows 11                  |
| Python      | 3.10.18                     |
| torch       | 2.2.1+cu118                 |
| torchvision | 0.17.1+cu118                |
| torchaudio  | 2.2.1+cu118                 |
| CPU         | Intel i7-12800HX            |
| GPU         | NVIDIA RTX 4070 Laptop  |
| RAM         | 32 GB            |

Edge Device Nvidia Jetson Orin Nano Super Information：

| Component   | Version/Spec                |
| ----------- | --------------------------- |
| GPU          |NVIDIA Ampere (512 CUDA cores16 Tensor cores)                  |
| CPU      |Cortex-A78AE                   |
| RAM       | 4GB 64bit LPDDR5               |
| Storage | 256GB NVME SSD                |
| OS  | Ubuntu22.04 LTS                 |
| Python         | 3.10.12           |
| PyQT5         | 5.15.6 |
| TensorRT         | 10.7.0            |

If you wish to reproduce the same environment, it is recommended to use **Conda** for management.
 Make sure you have [Anaconda](https://www.anaconda.com/download) installed, then run:

```
(base): conda create -n zhjoyX python=3.10.18 -y

# If you are in mainland China, add Tsinghua mirror to improve download speed:
# conda create -n zhjoyX python=3.10.18 -y -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --trusted-host https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# You can also set it globally:
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Activate the environment:

```
(base): conda activate zhjoyX
(zhjoyX): pip install -r requirements.txt ./
```
By the way ,if U wanna get the same environment on jetson nano edge device with linux OS,pls activate your Virtual Environment and try command :
```
pip install -r jetson_requirements.txt ./
```
------
> requirements.txt uploaded already !🤷‍♂️

## 🏋️ 2. Training the Model

1. Prepare your **YOLO-format annotated dataset**.
2. Integrate the improved modules from this repo into the **YOLOv12** open-source code.
3. Modify the configuration files accordingly and start training.

------

## 📚 3. References

> Core module papers, code repositories, experiment baselines, and lightweight model references.

**① Improved modules:**

- [PSConv](https://github.com/JN-Yang/PConv-SDloss-Data)
- [EUCB](https://github.com/SLDGroup/EMCAD)
- [MSEE](https://github.com/BellyBeauty/MDSAM)

**② Baseline comparison models:**

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main)
- [YOLOv13](https://github.com/iMoonLab/yolov13)
- [YOLOv8 / YOLOv10 / YOLOv11](https://github.com/ultralytics/ultralytics)

**③ Lightweight techniques:**

- [Torch-Pruning (TP)](https://github.com/VainF/Torch-Pruning)
- [LAMP](https://github.com/jaeho-lee/layer-adaptive-sparsity)
- [BCKD](https://github.com/TinyTigerPan/BCKD)
- [CWD](https://git.io/Distiller)
- [ICKD](https://github.com/ADLab-AutoDrive/ICKD)
- [MGD](https://github.com/yzd-v/MGD)

------



## 📬 Contact

If you have any questions about this repository or the related paper, feel free to reach out:

- 🚀 QQ: **2422785900**
- 🤖 Email: **zhj0109@st.gxu.edu.cn**

## Star History
> if U like this repo，pls give me a star ! thk u very much 😁
<a href="https://www.star-history.com/#zhanghjoy/Star_Fruit_Detection&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=zhanghjoy/Star_Fruit_Detection&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=zhanghjoy/Star_Fruit_Detection&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=zhanghjoy/Star_Fruit_Detection&type=Date" />
 </picture>
</a>
