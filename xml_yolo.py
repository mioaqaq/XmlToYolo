from tqdm import tqdm
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# 配置参数集中管理
CONFIG = {
    "image_sets": ["train", "val","trainval"],
    "path": Path("E:/bangong/yolo_test/datasets/VOC2028/"),     #你的xml数据集位置
    "class_names": ["hat", "person"],  # 示例类名，请根据你实际标注改
    "copy_images": True,
}

# 将VOC格式标注转换为YOLO格式
def convert_label(path, lb_path, image_id, names):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    try:
        with open(path / f'Annotations/{image_id}.xml', encoding='utf-8') as in_file, \
                open(lb_path, 'w', encoding='utf-8') as out_file:
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in names:
                    print(f"[WARNING] Unknown class '{cls}' in {image_id}.xml")
                    continue
                xmlbox = obj.find('bndbox')
                bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                cls_id = names.index(cls)
                out_file.write(" ".join(str(round(a, 6)) for a in (cls_id, *bb)) + '\n')
    except Exception as e:
        print(f"[ERROR] Failed to process {image_id}: {e}")

# 主处理函数
def prepare_voc_dataset(config):
    path = config["path"]
    class_names = config["class_names"]
    copy_images = config["copy_images"]

    for image_set in config["image_sets"]:
        imgs_path = path / 'images' / image_set
        lbs_path = path / 'labels' / image_set
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)

        txt_file = path / f'ImageSets/Main/{image_set}.txt'
        if not txt_file.exists():
            print(f"[ERROR] {txt_file} does not exist.")
            continue

        with open(txt_file) as f:
            image_ids = f.read().strip().split()

        for image_id in tqdm(image_ids, desc=f'Processing {image_set}'):
            img_src = path / f'JPEGImages/{image_id}.jpg'
            if not img_src.exists():
                print(f"[WARNING] Image {img_src} not found, skipping.")
                continue

            img_dst = imgs_path / img_src.name
            if copy_images:
                shutil.copyfile(img_src, img_dst)

            lb_path = (lbs_path / img_src.name).with_suffix('.txt')
            convert_label(path, lb_path, image_id, class_names)

# 程序入口
if __name__ == "__main__":
    prepare_voc_dataset(CONFIG)
