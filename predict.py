import os
import time
import argparse
import paddle
from PIL import Image
import numpy as np
import paddle.nn.functional as F
from cyclemlp import build_cyclemlp as build_model
from config import get_config
from config import get_pred_config
from datasets import get_val_transforms
import matplotlib.pyplot as plt
import label_list

def get_arguments():
    parser = argparse.ArgumentParser('predict')
    parser.add_argument('-cfg', type=str, default=None, help='the model config file')
    parser.add_argument('-image_path', type=str, default=None, help="the predict image, path")
    parser.add_argument('-pretrained', type=str, default=None, help="the pretrained model path")
    arguments = parser.parse_args()
    return arguments

@paddle.no_grad()
def predict(config):
    # build model
    model = build_model(config)
    model.eval()

    # load pretrained
    print(config.MODEL.PRETRAINED + '.pdparams')
    assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
    state_dict = paddle.load(config.MODEL.PRETRAINED + '.pdparams')
    model.set_dict(state_dict)

    # load the image and transform the image
    assert os.path.isfile(config.IMAGE_PATH) is True
    val_transformer = get_val_transforms(config)
    image = np.array(Image.open(config.IMAGE_PATH).convert("RGB"))
    img = val_transformer(image)
    img = img.unsqueeze(0)
    
    # start pred
    st_time = time.time()
    pred = F.softmax(model(img)).numpy()[0]
    end_time = time.time()

    label = pred.argmax()
    prob = pred[label]
    return label, prob, image, end_time-st_time

def main():
    arguments = get_arguments()
    config = get_config()
    config = get_pred_config(config, arguments)
    class_id, prob, image, pred_time = predict(config)
    plt.imshow(image)
    print(f"class_id:{label_list.label_dict[class_id]}, prob:{prob}, cost time:{pred_time:.4f}")
    plt.show()

if __name__ == "__main__":
    main()