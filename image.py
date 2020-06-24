# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from PIL import Image


 
from numpy import sin,pi

def solve(box):
     x1,y1,x2,y2,x3,y3,x4,y4= box[:8]
     cx = (x1+x3+x2+x4)/4.0
     cy = (y1+y3+y4+y2)/4.0  
     w = (np.sqrt((x2-x1)**2+(y2-y1)**2)+np.sqrt((x3-x4)**2+(y3-y4)**2))/2
     h = (np.sqrt((x2-x3)**2+(y2-y3)**2)+np.sqrt((x1-x4)**2+(y1-y4)**2))/2   

     sinA = (h*(x1-cx)-w*(y1 -cy))*1.0/(h*h+w*w)*2
     angle = np.arcsin(sinA)
     return angle,w,h
                              
def rotate_cut_img(im,degree,box,w,h,leftAdjust=True,rightAdjust=True,alph=0.2):
    x1,y1,x2,y2,x3,y3,x4,y4 = box[:8]
    x_center,y_center = np.mean([x1,x2,x3,x4]),np.mean([y1,y2,y3,y4])
    degree_ = degree*180.0/np.pi
    right = 0
    left  = 0
    if rightAdjust:
        right = 1
    if leftAdjust:
        left  = 1
    
    box = (max(1,x_center-w/2-left*alph*(w/2))##xmin
           ,y_center-h/2-left*alph*h,##ymin
           min(x_center+w/2+right*alph*(w/2),im.size[0]-1)##xmax
           ,y_center+h/2+left*alph*h)##ymax
    newW = box[2]-box[0]
    newH = box[3]-box[1]
    tmpImg = im.crop(box)
    return tmpImg,newW,newH

def sort_box(box):
    newbox = sorted(box,key=lambda x:((sum([x[1],x[3],x[5],x[7]])//4)//10,(sum([x[0],x[2],x[4],x[6]])//4)//10))
    return list(newbox)
