# 🚀 A Lightweight Star Fruit Quality Detector Based on a Label-Balanced Strategy and Deployed on Real-Time Edge Devices 🚀

👨‍💻 Author: [Haijun Zhang](https://github.com/zhanghjoy)

------

## 📌 Updates Record

- ✅ **2025/07/11** — Build this repo!
- ✅ **2025/09/02** — Updated utility files: `Star_Fruit_Detection/src/utils`
- ✅ **2025/09/16** — Updated modified modules and network YAML configuration files

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
| torch       | pls fill me🤨             |
| torchvision | pls fill me🤨             |
| torchaudio  | pls fill me🤨              |
| CPU         | Intel i7-12800HX            |
| GPU         | NVIDIA RTX 4070 Laptop (8G) |
| RAM         | 32 GB (4800 MHz)            |

If you wish to reproduce the same environment, it is recommended to use **Conda** for management.
 Make sure you have Anaconda installed, then run:

```
(base): conda create -n zhjoyX python=3.10.18 -y

# If you are in mainland China, add Tsinghua mirror to improve download speed:
# -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --trusted-host https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# You can also set it globally:
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Activate the environment:

```
(base): conda activate zhjoyX
(zhjoyX): pip install -r requirements.txt ./
```

------

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
- [SSD](https://github.com/bubbliiiing/ssd-pytorch)
- [Faster-RCNN](https://github.com/bubbliiiing/faster-rcnn-pytorch)
- [YOLOv13](https://github.com/iMoonLab/yolov13)
- [YOLOv8 / YOLOv10 / YOLOv11](https://github.com/ultralytics/ultralytics)

**③ Lightweight techniques:**

- [Torch-Pruning (TP)](https://github.com/VainF/Torch-Pruning)

------



## 📬 Contact

If you have any questions about this repository or the related paper, feel free to reach out:

- 🚀 QQ: **2422785900**
- 🤖 Email: **zhj0109@st.gxu.edu.cn**
