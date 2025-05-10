# XmlToYolo
xml转yolo格式脚本
##xml格式
dataset/
├── Annotations/           # 存放所有 XML 标注文件
│   ├── people(1).xml
│   ├── people(2).xml
│   └── ...
├── JPEGImages/            # 存放所有图像文件（.jpg/.png）
│   ├── people(1).jpg
│   ├── people(2).jpg
│   └── ...
├── ImageSets/
│   └── Main/
│       ├── train.txt      # 只包含图片名（无扩展名）
│       └── val.txt
└── voc.yaml               # 可选，转换工具需要时用

##yolo格式
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
