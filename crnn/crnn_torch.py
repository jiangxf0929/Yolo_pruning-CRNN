#coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable 
from crnn.utils import strLabelConverter,resizeNormalize
from crnn.network_torch import CRNN
from crnn import keys
from collections import OrderedDict
from config import ocrModel

def crnnSource():
    """
    加载模型
    """
    alphabet = keys.alphabetEnglish##英文模型
    converter = strLabelConverter(alphabet)

    model = CRNN(32, 1, len(alphabet)+1, 256, 1)
    
    trainWeights = torch.load(ocrModel,map_location=lambda storage, loc: storage)
    modelWeights = OrderedDict()
    for k, v in trainWeights.items():
        name = k.replace('module.','')
        modelWeights[name] = v
    model.load_state_dict(modelWeights)
    return model,converter

##加载模型
model,converter = crnnSource()
model.eval()
def crnnOcr(image):
       """
       crnn模型，ocr识别
       image:PIL.Image.convert("L")
       """
       scale = image.size[1]*1.0 / 32

       w = image.size[0] / scale

       w = int(w)
       transformer = resizeNormalize((w, 32))

       image = transformer(image)

       image = image.astype(np.float32)
       image = torch.from_numpy(image)
       image       = image.view(1,1, *image.size())
       image       = Variable(image)
       preds       = model(image)
       #取5530类中，概率最大的类别[16,1,5530]->[16,1]
       _, preds    = preds.max(2)
       preds       = preds.transpose(1, 0).contiguous().view(-1)
       sim_pred    = converter.decode(preds)
       return sim_pred