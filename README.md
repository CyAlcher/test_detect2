# 3DAnomalyDetect

The 3D Printing Anomaly Detection Algorithm project primarily aims to automatically identify various anomalies during the 3D printing process, including phenomena such as spaghetti, clogging, and layer shifting. The project is dedicated to continuously refining the definition standards for anomaly labels and enhancing the algorithm's mean Average Precision (mAP) for detection. The challenge lies in the comprehensive, timely, and accurate collection of data and the detection of anomalies as they occur.

> Keywords:
> FDM、Gantry

# Getting Started

## System Requirements

* Ubuntu >= `22.0`

## Installation

To install the dependencies, run:

```
sudo apt-get install ffmpeg
conda create -n detect python=3.8
pip install -r requirements.txt
```

## Object Labeling

Object labeling is an important task involving the process of identifying and marking objects in images, video frames, or other visual data. This process is essential for training machine learning models to recognize and classify different objects. The labeling methods table is as follows:

| Abnormal Types | Translate | Labeling Methods                                       |
|------------|----------------------|----------------------|
| Spaghetti  | 炒面     | X-AnyLabeling |
| Clogging    | 堵塞 |  X-AnyLabeling(collect normal samples) & Label-Start-End |
| Layer Shifting   | 丢步 | X-AnyLabeling(collect normal samples) & Label-Start-End |


The main steps of labeling are as follows:

- Read the video to generate frames to the data/frames folder

- Select the method based on the type of anomaly

- Generate labels files to the data/labels folder

- Complete the missing frame labels through a script, merge different labels

- Enhance the data, including rotation, grayscale adjustment, etc.

The existing problems:

- The occurrence of anomalies lacks a strict definition of the start time

- The duration of anomalies lacks a strict definition

- Reliance on individual experience for judgment can lead to biases in labeling samples

To mitigate these problems, for anomaly classification, we set three meta classes:

- normal
- possible anomaly
- (confirmed)anomaly

The labels are as follows:

| label_id | label_name |
| --- | --- |
|0 |normal |
|1 | possible-spaghetti |
|2 | spaghetti |
|3 | possible-clogging |
|4 | clogging |
|5 | possible-layershift |
|6 | layershift |

### Label 1: X-AnyLabeling

The tool [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) is powerful to mark objects. Some reasons for using the tool:

**1. Automation**: Object labeling tools can automate the manual labeling process, saving a significant amount of time and labor.

**2. Accuracy**: These tools often use advanced algorithms to improve the accuracy of labeling and reduce human errors.

**3. Scalability**: They can handle large datasets, making it feasible to perform object labeling in large-scale projects.

**4. Consistency**: Object labeling tools ensure that all labels follow the same standards and rules, thus improving the consistency of the dataset.

For the tutorial, you can refer to the [Bili](https://www.bilibili.com/video/BV1xt421J7VZ/?spm_id_from=333.337.search-card.all.click&vd_source=993a7ecc2ade79ea4ddfd2e56f597f4d).

Configs:

* The model of X-AnyLabeling is `Segment Anything (ViT-Large)`
* The classes(see data/classes.txt) are as follows:
```
normal
possible-spaghetti
spaghetti
possible-clogging
clogging
possible-layershift
layershift
```
* The export format is `YOLO Annotations` 

The output file of some frame is like `frame_0001.txt`, and the context is
```
{Label} {The horizontal coordinate of the boundary point 1} {The vertical coordinate of the boundary point 1} ......
```

For example
```
0 0.36328125 0.3515625 0.36328125 0.35546875
```


### Label 2: Label-Start-End

Label the start and end frames of the anomaly. The output file of some frame is also like `frame_0001.txt`. The context  is
```
{Label}
```

## Data Process

```
python process.py
```

## Model Train

```
python detect.py
```

# Advanced Materials

The 3D printing process can be referred to videos on [Bilibili](https://www.bilibili.com/video/BV12N411k7Wh/?spm_id_from=333.337.search-card.all.click&vd_source=90798ef7dc4ceb389931cd299ca22633), where [bambu](https://bambulab.cn/zh-cn/x1) has accumulated a lot of experience in the 3D printing industry for reference.


## 3D-Print

3D printers, also known as three-dimensional printers, are devices that construct three-dimensional objects layer by layer based on digital model files (such as STL or OBJ formats). They have a wide range of applications in fields like industrial design, architecture, medical, education, and art. 

### Common Types

The types include single-arm, dual-arm, box-type, and gantry—are common types of 3D printers. Here's a more detailed introduction to each:

**Single-Arm 3D Printers**

These printers typically have a mechanical arm that moves the print head over the printing platform, building the object layer by layer. Single-arm printers are the most common type and are suitable for personal use and small-scale production.

**Dual-Arm 3D Printers**

Dual-arm printers have two print heads, allowing them to print simultaneously, which can increase printing efficiency.

The two print heads can also perform different tasks, such as one printing support structures while the other prints the main object.

**Box-Type 3D Printers**

Box-type printers usually have an enclosed printing space, shaped like a box, which provides better control over the printing environment and reduces interference during the printing process. This design helps improve print quality, especially when printing materials that are sensitive to the environment.

**Gantry 3D Printers**

Gantry printers have a structure similar to a gantry crane, with the printing platform located underneath the gantry, and the print head moving on the gantry. This design allows the printer to have a larger build volume, making it suitable for printing large objects.

### Technologies Types

In addition to these types, 3D printers can also be classified according to different printing technologies, such as:

**Fused Deposition Modeling (FDM)**: The most common technology, which uses melted plastic filament material to stack layers.

**Stereolithography (SLA)**: Uses photopolymer resin that is cured layer by layer with ultraviolet light.

**Selective Laser Sintering (SLS)**: Uses a laser to melt and fuse powder materials together.

**Digital Light Processing (DLP)**: Similar to SLA but uses different light sources and projection techniques.

Each technology has its specific advantages and limitations and is suitable for different application scenarios. For example, FDM printers are widely popular due to their low cost and ease of use, while SLA and DLP printers are often used for precision manufacturing and prototype production because they can print finer details. SLS technology is suitable for printing complex geometric shapes because it can use powder materials without the need for support structures.

### Abnormal Types

| Abnormal Types | Translate | Image                                       |
|------------|----------------------|----------------------|
| Spaghetti  | 炒面     | <img src="https://cdn1.bambulab.com/zh/x1/ai_first_layer_fail-v1.png" width="200" height="100">|
| Clogging    | 堵塞 |     |
| Layer Shifting   | 丢步 |    |

## Algorithms

| Keyworkds  | Describe                                       |
|------------|----------------------|
|[yolo detect](https://www.liebertpub.com/doi/epdf/10.1089/3dp.2021.0231)| The evaluation of the models showed promising results in classifying defects quickly and accurately. The optimized models (YOLOv3-Tiny 100 and 300 epochs) achieved a mean average precision score of >80% using the AP50 metric and an inference speed of 70 frames per second.
|[ViT](https://github.com/lucidrains/vit-pytorch) | A simple way to achieve SOTA in vision classification with only a single transformer encoder. |

The [doc](https://docs.ultralytics.com/modes/predict/#obb) for yolo.