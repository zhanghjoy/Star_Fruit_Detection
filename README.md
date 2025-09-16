🚀 基于标签均衡策略的轻量化杨桃品质检测器，并部署于实时边缘设备 🚀  
[Haijun Zhang](https://github.com/zhanghjoy)  

---



# 📌 更新记录

* :white_check_mark: 2025/07/11 创建本仓库！
* :white_check_mark: 2025/09/02 更新工具文件：Star_Fruit_Detection/src/utils
* :white_check_mark: 2025/09/02 更新修改后的模块与网络配置文件  

---

# 📖 摘要

本仓库用于存放与论文相关的代码、工具以及参考资料。  

---

# 0️⃣ 前言

- **src**：存放实验中使用的工具脚本，包括元素库均衡代码、图像分割代码、数据增强脚本、标签统计工具、改进模块代码以及配置文件。  
- **photo**：存放已处理完成的元素库图像。  
- **开源数据集**：[Mendeley Dataset](https://data.mendeley.com/datasets/f35jp46gms/1)  
- **标注工具**：[labelImg](https://github.com/tzutalin/labelImg)  

---

# 1️⃣ 环境配置

本文实验环境如下：  

| 环境项          | 版本信息                  |
| --------------- | ------------------------- |
| 操作系统        | Windows 11                |
| Python版本      | 3.10.18                   |
| torch版本       | Fill me                          |
| torchvision版本 | Fill me                          |
| torchaudio版本  | Fill me                           |
| CPU             | Intel i7-12800HX          |
| GPU             | NVIDIA RTX4070 Laptop(8G) |
| 内存            | 32G (4800MHz)             |

如果你希望复现相同的软件环境，推荐使用 **Conda** 进行管理。在确保已安装 **Anaconda** 的前提下运行以下命令：  

```bash
(base): conda create -n zhjoyX python=3.10.18 -y
# 如果你在中国国内，建议使用清华源镜像以提升下载速度：
# -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --trusted-host mirrors.tuna.tsinghua.edu.cn
# 也可以全局设置清华源：
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

激活环境：  

```bash
(base): conda activate zhjoyX
(zhjoyX): pip install -r requirements.txt
```

---

# 2️⃣ 代码训练

1. 准备好 **YOLO格式** 并完成标注的数据集。  
2. 将本仓库提供的改进模块添加到 YOLOv12 的开源代码中。  
3. 根据配置文件进行训练。  

---

# 3️⃣ 参考资料

📑 本研究所涉及的核心模块相关论文、代码仓库，以及对比实验与轻量化工具参考如下：  

**1. 改进模块参考仓库**  

* [PSConv](https://github.com/JN-Yang/PConv-SDloss-Data)  
* [EUCB](https://github.com/SLDGroup/EMCAD)  
* [MSEE](https://github.com/BellyBeauty/MDSAM)  

**2. 对比实验模型仓库**  

* [RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main)  
* [SSD](https://github.com/bubbliiiing/ssd-pytorch)  
* [Faster-RCNN](https://github.com/bubbliiiing/faster-rcnn-pytorch)  
* [YOLOv13](https://github.com/iMoonLab/yolov13)  
* [YOLOv8/YOLOv10/YOLOv11](https://github.com/ultralytics/ultralytics)  

**3. 轻量化参考工具**  

* [Torch-Pruning TP](https://github.com/VainF/Torch-Pruning)  

---

# 📬 联系方式

如果你对本仓库或论文有任何疑问，欢迎联系我，我将尽力为你解答：  

:rocket: QQ账号：2422785900  
:robot: 邮箱：zhj0109@st.gxu.edu.cn  
