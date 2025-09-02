<div align="center">
<h1>A Lightweight Star Fruit Quality Detector Based on a Label-Balanced Strategy and Deployed on Real-Time Edge Devices	</h1>

[Haijun Zhang](https://github.com/zhanghjoy)<sup>1</sup>, [Jiqin Chen](https://prof.gxu.edu.cn/teacherDetails/20461de7-0943-4b47-85f1-015f72dcb8d3)<sup>1</sup>

<sup>1</sup>  Guangxi University

---

# Updates Record

* 2025/09/02 Buid this repo
* liu :()

---

# Abstract

这个仓库用于存放论文相关的代码和工具以及参考资料。



---

# 0.前言

本文使用的开源工具和代码地址：

labelimg：

yolov12：

开源数据集：

本仓库对应的对应的论文为：

RTDETR

SSD FasterRCNN (bubling)

仓库的结构为：

photos为

Tools_Src：存放文中使用的一些工具脚本，元素库平衡代码，分割代码，背景去除代码。图像数据增强代码

ultralytics：存放yolov12的官方文件

zhjoy_cfg: 存放使用到的自定义的yolo配置文件

>

# 1.环境配置

本文的训练环境为：

| 环境            | 版本                      |
| --------------- | ------------------------- |
| 操作系统        | windows11                 |
| 语言版本        | 3.10.18                   |
| torch版本       |                           |
| torchvision版本 |                           |
| torchaudio版本  |                           |
| CPU             | Intel i7-12800HX          |
| 显卡            | NVIDIA RTX4070 Laptop(8G) |
| 内存            | 32G(4800MHz)              |
|                 |                           |

如果你想得到相同的软件实验环境，推荐使用Conda对环境进行管理，在确保你的电脑已经安装了Anaconda软件的前提下运行以下命令：

```bash
(base):conda create -n zhjoyX python=3.10.18 -y
# 如果你在中国国内，推荐加上清华源镜像能够使你的下载更加流畅：
# -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --trusted-host=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
#此外，你还可以将其全局设置为清华源：
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

环境激活

```bash
(base):conda activate zhjoyX
(zhjoyX):pip install -r requirments.txt./
```

# 2.代码训练

首先你需要准备自己的yolo格式已经标注完成的数据集

# 3.参考资料

> 核心模块相关论文和代码仓库：
>
> 

1.论文中对比实验的参考代码





---

如果对本仓库或者论文有疑问，可以通过以下方式联系我，我将尽最大的努力为您解答。

:rocket: 账号(QQ):2422785900

:robot: 邮箱(Email):zhj0109@st.gxu.edu.cn

