import tensorflow as tf

class GeneralizedRCNNTransform():
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        # 最小边长范围
        self.min_size = min_size
        # 最大边长范围
        self.max_size = max_size
        # 标准化处理的均值
        self.image_mean = image_mean
        # 标准化处理的标准差
        self.image_std = image_std

    def normalize(self, image):
        """
        标准化处理
        :param image:
        :return:
        """
        dtype, device = image.dtype, image.device
        mean = tf.convert_to_tensor(self.image_mean, dtype=dtype, device=device)
        std = tf.convert_to_tensor(self.image_std, dtype=dtype, device=device)
        # 对 mean\std 进行维度扩展，[3] -> [3 , 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def tf_choice(self, k):
        # type: (List[int]) -> int
        index = int(tf.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        获取缩放因子，缩放图片和bounding box大小
        :param image:
        :param target:
        :return:
        """
        # image.shape [channel, height, width]
        h, w = image.shape[-2:]
        im_shape = tf.convert_to_tensor(image.shape[-2:])
        # 获取高宽的最小值
        min_size = float(tf.reduce_min(im_shape))
        max_size = float(tf.reduce_max(im_shape))

        if self.training:
            # 指定输入图片的最小边长
            size = float(self.tf_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        # 缩放系数
        scale_factor = size / min_size

        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size

        # 插值缩放图片
        image = tf.image.resize(image[None], scale_factor=scale_factor, method="ResizeMethod.BILINEAR")[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        # 同比缩放bbox
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def batch_images(self, images, size_divisible=32):
        """
        图片批处理
        :param images:
        :param size_divisible: 将图片宽高调整为该数的整数倍
        :return:
        """

    def postprocess(self,
                    result,                # type: List[Dict[str, Tensor]]
                    image_shapes,          # type: List[Tuple[int, int]]
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        """
        将bbox还原到图像上
        :param result:
        :param image_shapes:
        :param original_image_sizes:
        :return:
        """



def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    缩放bbox
    :param boxes:
    :param original_size: 图像缩放前的尺寸
    :param new_size: 图像缩放后的尺寸
    :return:
    """
    ratios = [
        tf.tensor(s, dtype=tf.float32, device=boxes.device) /
        tf.tensor(s_orig, dtype=tf.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return tf.stack((xmin, ymin, xmax, ymax), dim=1) # 在维度1上进行合并



