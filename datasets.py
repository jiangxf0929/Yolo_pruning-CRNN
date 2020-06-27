import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset

from utils.utils import xyxy2xywh, xywh2xyxy

vid_formats = ['.mov', '.avi', '.mp4']

def diffimage(src_1,src_2):#帧差法
    #src_1=np.array(src_1)
    #src_2=np.array(src_2)
    src_1 = src_1.astype(np.int)
    src_2 = src_2.astype(np.int)
    diff = abs(src_1 - src_2)
    return diff.astype(np.uint8)

class LoadImages:  # for inference
    
    def __init__(self, path, img_size=416):#传入path和图片大小
        path = str(Path(path))  # os-agnostic
        self.img_size = img_size
        self.path_name=path
        self.nF = 1  
        self.new_video(path)

    def tiaozhen(self):
        jump_frame=5#每次读取到图片进行跳帧
        for index in range(jump_frame):
            ret_val, frame_new = self.cap.read()
            self.frame += 1#读到当前帧数,记录的帧数加1
            if not ret_val:#没读取到视频
                self.cap.release()#关闭视频
                raise StopIteration

    def xunhuan(self):#定义提取帧的循环
        self.lastframe=[]#声明一个空数组
        self.yuzhi = 15#帧提取法阈值
        self.timeF = 11#每隔多少帧读取三个
        crop_rate_width=0.94#视频横向裁剪比
        crop_rate_height=0.94#视频纵向裁剪比
        for index in range (self.timeF):#一次性读取self.timeF帧
            ret_val, frame_new = self.cap.read()
            self.frame += 1#读到当前帧数,记录的帧数加1
            if not ret_val:#没读取到视频
                self.cap.release()#关闭视频
                raise StopIteration
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            x0=width*(1-crop_rate_width)/2
            y0=height*(1-crop_rate_height)/2
            x1=width-x0
            y1=height-y0
            try:
                frame_new=frame_new[int(y0):int(y1),int(x0):int(x1)]#裁剪图片
            except:
                raise StopIteration
            self.lastframe.append(frame_new)

        thr=np.sum(diffimage(self.lastframe[self.timeF-1],self.lastframe[(self.timeF//2)])+diffimage(self.lastframe[self.timeF//2],self.lastframe[0]))/self.lastframe[0].size#阈值的计算
        return thr

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):#含有__next__()函数的对象都是一个迭代器
        if self.count == self.nF:
            raise StopIteration
        path=self.path_name
        self.mode = 'video'
        thr=self.xunhuan()
        while thr>=0:
            if  thr<=self.yuzhi:
                img0=self.lastframe[(self.timeF//2)]
                self.keyframe+=1
                #self.tiaozhen()#是否跳帧
                break
            if thr>self.yuzhi:
                thr=self.xunhuan()


        print('video %g/%g (%g/%g) %s keyframe_num %g : ' % (self.count+1, self.nF, self.frame, self.nframes, path,self.keyframe), end='')
        #如video 1/1 （2/7200）xxx.mp4字样
        
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert转换
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):#读取新视频
        self.frame = 0
        self.keyframe=0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))#总帧数

    def __len__(self):
        return self.nF  # 文件个数

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


