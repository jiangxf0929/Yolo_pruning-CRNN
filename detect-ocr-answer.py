import argparse
import numpy as np
import pandas as pd
from sys import platform

from torch.autograd import Variable
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from image import union_rbox,adjust_box_to_origin
from crnn.crnn_torch import crnnOcr as crnnOcr
from image import  solve,rotate_cut_img,sort_box,get_boxes,letterbox_image

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

def text_take(text,ques):
    text_list=[]
    flag_=False
    for spr in text:
        if spr.find(ques) is not -1:
            flag_=True
            str_list=spr.split(ques)
            for i in str_list:
                if i is '':
                    continue
                text_list.append(i)
        else:
            text_list.append(spr)
    return text_list,flag_


def detect(save_img=False):
    img_size = (320,192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    output=[]
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = 'output'#str(Path(out) / Path(p).name)
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
                    
                newbox = sort_box(Box)
                result = crnnRec(im0*255,newbox,True,True,0.05,1.0)
                output.append(result)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

    '''
    answer
    read in  questions
    '''
    question=opt.input
    q=pd.read_csv(question,header=None,sep=';')
    question=np.array(q)

    for ques in question[0,]:
        exist=False
        for text in output:
            flag=False
            text_list=[]
            text_list,flag=text_take(text,ques)
            if flag==True:
                with open('answer.txt','a') as file:
                    file.write(('%s'+':')%ques)
                for i in text_list[0:-1]:
                    with open('answer.txt','a') as file:
                        file.write(('%s'+' ')%i)
                with open('answer.txt','a') as file:
                    file.write(('%s'+';')%text_list[-1])
                exist=True
            else :
                continue
        if exist is False:
            with open('answer.txt','a') as file:
                file.write(('%s'+':-;')%ques) 
                
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/prune_0.8_keep_0.1_8_shortcut_yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/text.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/last.weights', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--input', type=str, default='question.txt', help='input folder')
    parser.add_argument('--output', type=str, default='output', help='output folder')# output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
