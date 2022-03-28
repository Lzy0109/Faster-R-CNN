import json
import os
from PIL import Image
from lxml import etree
import tensorflow as tf

class VOCDataSet(object):
    """
    PASCAL VOC数据集类
    """
    def __init__(self, transforms=None):
        # 文件路径
        self.root = os.path.join()
        self.img_root = os.path.join()
        self.annotations_root = os.path.join()
        self.transforms = transforms

        # 读取train.txt val.txt
        txt_path = os.path.join()
        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0]

        # 检查文件
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

    def len(self):
        return len(self.xml_list)

    def getItem(self, idx):
        """
        读取xml文件 获取单个目标信息
        :param idx:
        :return:
        """
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["anntation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        # 格式检测
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []

        # bounding box属性
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # 转换为张量
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int64)
        iscrowd = tf.convert_to_tensor(iscrowd, dtype=tf.int64)
        image_id = tf.convert_to_tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxex"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 图像预处理
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_widtyh(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析为字典形式
        :param xml:
        :return:
        """
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            # 递归
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))