import argparse
import numpy as np

from torch.autograd import Variable
from models import *  
from utils.datasets import *
from utils.utils import *
from crnn.crnn_torch import crnnOcr as crnnOcr
from image import  solve,rotate_cut_img,sort_box,get_boxes,letterbox_image
from answer import output_answer

def crnnRec(im,boxes,leftAdjust=True,rightAdjust=True,alph=0.2,f=1.0):
   results = []
   im = Image.fromarray(im) 
   for index,box in enumerate(boxes):
       degree,w,h,cx,cy = solve(box)
       partImg,newW,newH = rotate_cut_img(im,degree,box,w,h,leftAdjust,rightAdjust,alph)
       text = crnnOcr(partImg.convert('L'))
       if text.strip()!=u'':
           results.append(text)
   return results

def detect():
    img_size = (320,192) 
    source, weights =  opt.source, opt.weights
    device ='cpu'
    model = Darknet(opt.cfg, img_size)
    load_darknet_weights(model, weights)
    model.to(device).eval()
    dataset = LoadImages(source, img_size=img_size)
    names = load_classes(opt.names)
    t0 = time.time()
    output=[]
    box_num=0
    text_list=[]
    sentence_list=[]
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time.time()
        pred = model(img, augment=opt.augment)[0]
        t2 = time.time()
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            save_path = 'output'#str(Path(out) / Path(p).name)##
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                Box=[]
                # Write results
                for *xyxy, conf, cls in det:
                    xy=[i.numpy()+0
                         for i in xyxy]
                    Box.append([xy[0],xy[1],xy[2],xy[1],xy[2],xy[3],xy[0],xy[3]])

                #如果图中超过两个box且和前一句差别一个box
                if len(Box)>=2 and abs(len(Box)-box_num)>=1:
                    newbox = sort_box(Box)#获取坐标
                    result = crnnRec(im0*255,newbox,True,True,0.05,1.0)
                    output.append(result)
                    for i in result:
                        with open(save_path + '/text.txt', 'a') as file:
                            file.write(('%s'  + ';') % (i))
                    with open(save_path + '/text.txt', 'a') as file:
                          file.write('\n')
                    file.close
                    result = " ".join(result)#用空格分开各result
                    sentence_list.append(result)#得到句子列表
                box_num=len(Box)#更新box数目
               
            print('%sDone. (%.3fs)' % (s, t2 - t1))
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/prune_0.8_keep_0.1_8_shortcut_yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/text.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/last.weights', help='weights path')
    parser.add_argument('--source', type=str, default='', help='source') 
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()
        output_answer("./output/text.txt")#输出答案