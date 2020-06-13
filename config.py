import os
########################文字检测########################
##文字检测引擎 
pwd = os.getcwd()

LSTMFLAG = True
##模型选择 True:中英文模型 False:英文模型
ocrFlag = 'torch'##ocr模型 支持 keras  torch版本
chinsesModel = True
##纯英文模型
if LSTMFLAG is True:
    ocrModel = os.path.join(pwd,"weights","ocr-english.pth")
else:
    ocrModel = os.path.join(pwd,"weights","ocr-lstm.pth")

######################OCR模型######################