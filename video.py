import argparse
import numpy as np
import time
import threading
from PIL import Image, ExifTags
from torch.autograd import Variable
from models import *  
from utils.utils import *
from crnn.crnn_torch import crnnOcr as crnnOcr
from image import  solve,rotate_cut_img,sort_box

sentence_list=[]


class Mythread(threading.Thread):
   def __init__(self,func,args,):
      threading.Thread.__init__(self)
      self.func = func
      self.args = args 
   def run(self):
      self.res=self.func(*self.args)
   def getResult(self):
      return self.res

def Ocr(img0,Box):
   newbox = sort_box(Box)#获取坐标
   t4=time.time()
   result = crnnRec(img0*255,newbox,True,True,0.05,1.0)
   result = " ".join(result)#用空格分开各result
   result=result.replace("*","").upper().replace("  "," ")
   #sentence_list.append(result)#得到句子列表
   t2 = time.time()
   print('CRNN. (%.3fs)' % (t2 - t4))
   return result
   
   
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

def diffimage(src_1,src_2):#帧差法
    src_1 = src_1.astype(np.int)
    src_2 = src_2.astype(np.int)
    diff = abs(src_1 - src_2)
    return diff.astype(np.uint8)

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232 将图像大小调整为32像素多矩形
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border添加边框
    return img, ratio, (dw, dh)


def detect(source):
    img_size = 512
    weights = 'Yolo_pruning-CRNN/weights/last.weights'
    device ='cpu'
    model = Darknet('Yolo_pruning-CRNN/cfg/prune_0.8_keep_0.1_8_shortcut_yolov3.cfg', img_size)
    load_darknet_weights(model, weights)
    model.to(device).eval()
    names = load_classes('Yolo_pruning-CRNN/data/text.names')
    t0 = time.time()
    #video
    threshold=15
    timeF = 11
    crop_rate_width=0.94
    crop_rate_height=0.94
    vidcap = cv2.VideoCapture(source)
    success=True
    cnt=0
    box_num=0
    threads = []
    while success:
        success,img = vidcap.read()
        if not success:
            break
        cnt += 1
        if (cnt-1)%timeF!=0:
            continue
        lastframe=[]
        for index in range (timeF):#一次性读取self.timeF帧
            ret_val, frame_new = vidcap.read() 
            if ret_val == False:
                break    
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            x0=width*(1-crop_rate_width)/2
            y0=height*(1-crop_rate_height)/2
            x1=width-x0
            y1=height-y0
            frame_new=frame_new[int(y0):int(y1),int(x0):int(x1)]
            lastframe.append(frame_new)
        l=len(lastframe)
        thr=np.sum(diffimage(lastframe[l-1],lastframe[(l//2)])+diffimage(lastframe[l//2],lastframe[0]))/lastframe[0].size
        if thr>=threshold:
            continue
        img0 = lastframe[l//2]
        img = letterbox(img0,new_shape=img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time.time()
        pred = model(img)
        # Apply NMS
        conf_thres=0.1
        iou_thres=0.5
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        if pred==[None]:
            continue
        det=pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        Box=[]
        for *xyxy, conf, cls in det:
            xy=[i.detach().numpy()+0 for i in xyxy]
            Box.append([xy[0],xy[1],xy[2],xy[1],xy[2],xy[3],xy[0],xy[3]])
        t3=time.time()
        print('YoLo. (%.3fs)' % (t3 - t1))
        if len(Box)>=2 and abs(len(Box)-box_num)>=1:
            box_num=len(Box)#更新box数目 
            t = Mythread(Ocr,(img0,Box))
            threads.append(t)
        else:
            continue
    for i in range(len(threads)):
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()
        sentence_list.append(threads[i].getResult())  
    print('Done. (%.3fs)' % (time.time() - t0))