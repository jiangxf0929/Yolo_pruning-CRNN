import argparse
import numpy as np

from torch.autograd import Variable
from models import *  
from utils.datasets import *
from utils.utils import *
from crnn.crnn_torch import crnnOcr as crnnOcr
from image import  solve,rotate_cut_img,sort_box

def crnnRec(im,boxes,leftAdjust=True,rightAdjust=True,alph=0.2,f=1.0):
   results = []
   im = Image.fromarray(im) 
   for index,box in enumerate(boxes):
       degree,w,h = solve(box)
       partImg,newW,newH = rotate_cut_img(im,degree,box,w,h,leftAdjust,rightAdjust,alph)
       text = crnnOcr(partImg.convert('L'))
       if text.strip()!=u'':
           results.append(text)
   return results

def detect(source,question):
    img_size = 512
    weights = 'Yolo_pruning-CRNN/weights/last.weights'
    device ='cpu'
    model = Darknet('Yolo_pruning-CRNN/cfg/prune_0.8_keep_0.1_8_shortcut_yolov3.cfg', img_size)
    load_darknet_weights(model, weights)
    model.to(device).eval()
    dataset = LoadImages('Yolo_pruning-CRNN/'+source, img_size=img_size)
    names = load_classes('Yolo_pruning-CRNN/data/text.names')
    t0 = time.time()
    output=[]
    box_num=0
    text_list=[]
    sentence_list=[]
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time.time()
        pred = model(img)[0]
        t2 = time.time()
        # Apply NMS
        conf_thres=0.1
        iou_thres=0.5
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   multi_label=False)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                Box=[]
                # Write results
                for *xyxy, conf, cls in det:
                    xy=[i.detach().numpy()+0
                         for i in xyxy]
                    Box.append([xy[0],xy[1],xy[2],xy[1],xy[2],xy[3],xy[0],xy[3]])

                #如果图中超过两个box且和前一句差别一个box
                if len(Box)>=2 and abs(len(Box)-box_num)>=1:
                    newbox = sort_box(Box)#获取坐标
                    result = crnnRec(im0*255,newbox,True,True,0.05,1.0)
                    
                    result = " ".join(result)#用空格分开各result
                    result=result.replace("*","").upper().replace("  "," ")
                    sentence_list.append(result)#得到句子列表
                box_num=len(Box)#更新box数目               
            print('%sDone. (%.3fs)' % (s, t2 - t1))
    print('Done. (%.3fs)' % (time.time() - t0))
    return sentence_list
