#  A lightweight carambola detector based on label balancing and pruning-distillation 

👨‍💻 **Jiqing Chen**<sup>a,b</sup>, [**Haijun Zhang**<sup>a</sup>](https://github.com/zhanghjoy), **Yanyan Bao**<sup>a</sup>, **Jingyao Gai**<sup>a,b</sup>, **Qunchao He**<sup>a</sup>, **Anhong Ma**<sup>a</sup>, **Wei Wang**<sup>a</sup>

🏛️ <sup>a</sup> College of Mechatronic Engineering, Guangxi University, Nanning 530007, China

🏛️ <sup>b</sup> Guangxi Manufacturing System and Advanced Manufacturing Technology Key Laboratory, Nanning 530007, China

------

## 📌 Updates Record

- ✅ **2025/07/11** — Build this repo to record core file on paper.
- ✅ **2025/09/02** — Updated utility files: `Star_Fruit_Detection/src/utils`
- ✅ **2025/09/16** — Updated modified modules and network YAML configuration files.
- ✅ **2025/09/17** — Updated core pruning EXP code;Upload requirements.txt ;Create Repo License
- ✅ **2025/10/07** — The uploaded code is based on a PyQt5-developed UI interface, which supports selecting both .pt and .engine format weight files, as well as recognition in video, image, and camera mode.
- ✅ **2025/11/05** — The paper has been completed and submitted to the journal [Computers and Electronics in Agriculture](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture).
- 🪓 **2025/11/16** — The paper was rejected by journal [Computers and Electronics in Agriculture](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture).
- ✅ **2025/11/17** — The paper submitted to the journal [Engineering Applications of Artificial Intelligence](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence).
------

------

## Paper Status Record
1.[Engineering Applications of Artificial Intelligence](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence)
- ✅ 2025.11.17：Submitted to Journal
- 

------
## 📖 Abstract

Accurate detection and grading of star fruit are key to automated harvesting. However, object detection tasks often struggle to balance accuracy with model lightweighting. In grading tasks, class imbalance can significantly weaken the model’s learning ability and generalization performance. To address these issues, this study proposed a balancing strategy based on a label element set. This approach effectively alleviates class imbalance and improves the model’s cross-category detection capabilities. Using YOLOv12n as the baseline model, the PSConv and EUCB modules were introduced. The network structure was further optimized in combination with the MSEE and EDH modules proposed in this paper. These changes enhanced the model’s star fruit feature extraction abilities. To balance detection performance and model lightweighting, this paper also combines the LAMP pruning and BCKD knowledge distillation strategies. This achieves a strong balance between performance improvement and computational reduction. Experimental results show that the YOLOv12n-PDM model achieves an F1 score of 79.8% and an mAP of 84.5%. Compared to the baseline model, the proposed model reduces GFLOPS by 51.7% and parameters by 66.8%. Deployment experiments on a Jetson Orin Nano super edge computing device and a mobile robot platform demonstrate that the accelerated model achieves a real-time inference speed of 40.42 FPS. In summary, the proposed model outperforms existing mainstream object detection methods in detection accuracy, lightweightness, and real-time performance. It provides reliable technical support for the development of an intelligent star fruit picking and grading robot system.

We deployed this model on robot:
![20](https://github.com/user-attachments/assets/842b7d7c-fa99-4547-813e-0be155541d53)

work demo vedio:
https://github.com/user-attachments/assets/08a9dad1-e017-4f2d-a737-17dfe4401d20

---

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

- 🤖 Email: **zhanghjoy@163.com**

## Star History
> if U like this repo，pls give me a star ! thk u very much 😁
<a href="https://www.star-history.com/#zhanghjoy/Star_Fruit_Detection&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=zhanghjoy/Star_Fruit_Detection&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=zhanghjoy/Star_Fruit_Detection&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=zhanghjoy/Star_Fruit_Detection&type=Date" />
 </picture>
</a>
