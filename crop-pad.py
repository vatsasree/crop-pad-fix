import torch
import datetime
from doctr.models import detection
import numpy as np
from PIL import Image
# from torchvision.transforms import Normalize
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pypdfium2 as pdfium
from typing import Any
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from collections import OrderedDict
from doctr.utils.visualization import visualize_page
from datetime import date
import cv2
import os
import json
import re
import shutil
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage


aliases = {
    'model1': '/home2/sreevatsa/Robust-word-detector-for-Indic-Documents/weights/db_resnet50.pt',
    'model2': '/home2/sreevatsa/models/final/Class_balanced finetune for all_layers_epoch11.pt',
    'model3': '/home2/sreevatsa/models/final/Random_sampling-finetune for all layers (Backbone unfreezed)_epoch22.pt',
    'model4': '/home2/sreevatsa/models/final/Random_sampling-finetune for last layers (Backbone freezed)_v2_epoch9.pt'
}

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #changed
    # parser.add_argument("easy_train_path", type=str,default="/scratch/abhaynew/newfolder/train/Easy", help="path to training data folder")
    # parser.add_argument("medium_train_path", type=str,default="/scratch/abhaynew/newfolder/train/Medium", help="path to training data folder")
    # parser.add_argument("hard_train_path", type=str,default="/scratch/abhaynew/newfolder/train/Hard", help="path to training data folder")


    # parser.add_argument("val_path", type=str,default="/scratch/abhaynew/newfolder/val", help="path to validation data folder")
    # parser.add_argument("test_path", type=str,default="/scratch/abhaynew/newfolder/test", help="path to test data folder")
    parser.add_argument("--ImageFile", type=str, default=None, help="Input file to get text detections")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--input_size", type=int, default=1024, help="model input size, H = W")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=0, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, choices=aliases.keys(), metavar='choice',help="Path to your checkpoint")
    # parser.add_argument("--resume", type=str, default=None, choices=aliases.keys(), metavar='choice',help="Path to your checkpoint")
    # parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true", help="freeze model backbone for fine-tuning"
    )
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unormalized training samples"
    )
    parser.add_argument("--wb", dest="wb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", help="Push to Huggingface Hub")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load pretrained parameters before starting the training",
    )
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--padding',default=0,type=int,
                        help = 'amount of padding to bounding boxes (in pixels)')                    
    args = parser.parse_args()

    return args

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selection = args.resume
selected_model = aliases[args.resume]


if isinstance(selected_model, str):
    predictor = ocr_predictor(pretrained=True).to(device)
    state_dict = torch.load(selected_model)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    predictor.det_predictor.model.load_state_dict(new_state_dict)
else:
    predictor = ocr_predictor(pretrained=True).to(device)

today = date.today()
d=today.strftime("%d%m%y")

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%H%M%S")

def doctr_predictions_dir(directory): 
    doc = DocumentFile.from_images(directory)
    print('TPYRY')
    print(type(doc))
    print(np.shape(doc))
    print(doc)
    result = predictor(doc)
    dic = result.export()
    
    page_dims = [page['dimensions'] for page in dic['pages']]
    print(page_dims)
    regions = []
    abs_coords = []
    
    regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
    abs_coords = [
    [[int(round(word['geometry'][0][0] * dims[1])), 
      int(round(word['geometry'][0][1] * dims[0])), 
      int(round(word['geometry'][1][0] * dims[1])), 
      int(round(word['geometry'][1][1] * dims[0]))] for word in words]
    for words, dims in zip(regions, page_dims)
    ]

#     pred = torch.Tensor(abs_coords[0])
    # return (abs_coords,page_dims,regions)
    return abs_coords


def doctr_predictions_dir_ratios(directory): 
    doc = DocumentFile.from_images(directory)
    print('TPYRY')
    print(type(doc))
    print(doc)
    result = predictor(doc)
    dic = result.export()
    
    page_dims = [page['dimensions'] for page in dic['pages']]
    print(page_dims)
    regions = []
    abs_coords = []
    
    regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
    abs_coords = [
    [[(word['geometry'][0][0] ), 
      (word['geometry'][0][1] ), 
      (word['geometry'][1][0] ), 
      (word['geometry'][1][1] )] for word in words]
    for words, dims in zip(regions, page_dims)
    ]

#     pred = torch.Tensor(abs_coords[0])
    # return (abs_coords,page_dims,regions)
    return abs_coords

def rescaled_bboxes_from_cropped(img_cropped,img_source,top_left):
    left = top_left[0]
    top = top_left[1]
    
    img_source = cv2.cvtColor(cv2.imread(img_source),cv2.COLOR_BGR2RGB)
    target_h = img_source.shape[0]
    target_w = img_source.shape[1]
    # doc = DocumentFile.from_images(directory_cropped)
    # print('TPYRY')
    # print(type(doc))
    # print(doc)
    # print(type(img_cropped))
    img_cropp=[]
    img_cropp.append(img_cropped)
    # print(type(img_cropp))
    # print(np.shape(img_cropp))
    # print(img_cropp)
    result = predictor(img_cropp)
    dic = result.export()
    # print(dic)
    page_dims = [page['dimensions'] for page in dic['pages']]
    print(page_dims)
    regions = []
    abs_coords = []
    
    regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
    # abs_coords = [
    # [[int(round(word['geometry'][0][0] * target_w)), 
    #   int(round(word['geometry'][0][1] * target_h)), 
    #   int(round(word['geometry'][1][0] * target_w)), 
    #   int(round(word['geometry'][1][1] * target_h))] for word in words]
    # for words, dims in zip(regions, page_dims)
    # ]
    abs_coords = [
    [[int(round(word['geometry'][0][0] * dims[1]))+left, 
      int(round(word['geometry'][0][1] * dims[0]))+top, 
      int(round(word['geometry'][1][0] * dims[1]))+left, 
      int(round(word['geometry'][1][1] * dims[0]))+top] for word in words]
    for words, dims in zip(regions, page_dims)
    ]

#     pred = torch.Tensor(abs_coords[0])
    # return (abs_coords,page_dims,regions)
    return abs_coords

# def doctr_predictions(img): 
#     # doc = DocumentFile.from_images(directory)
#     img = list(img)
#     result = predictor(img)
#     dic = result.export()
    
#     page_dims = [page['dimensions'] for page in dic['pages']]
    
#     regions = []
#     abs_coords = []
    
#     regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
#     abs_coords = [
#     [[int(round(word['geometry'][0][0] * dims[1])), 
#       int(round(word['geometry'][0][1] * dims[0])), 
#       int(round(word['geometry'][1][0] * dims[1])), 
#       int(round(word['geometry'][1][1] * dims[0]))] for word in words]
#     for words, dims in zip(regions, page_dims)
#     ]

# #     pred = torch.Tensor(abs_coords[0])
#     # return (abs_coords,page_dims,regions)
#     return abs_coords



def visualize_preds_dir(img_dir):
    preds = doctr_predictions_dir(img_dir)
    img = cv2.cvtColor(cv2.imread(img_dir),cv2.COLOR_BGR2RGB)
    for w in preds[0]:
        cv2.rectangle(img,(w[0], w[1]),(w[2], w[3]),(0,0,255),1)
    # plt.imshow(img)
    cv2.imwrite('/home2/sreevatsa/output_test_doctrv2_{}_{}.png'.format(d,formatted_time), img)

# def visualize_preds(img):
#     preds = doctr_predictions(img)
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     for w in preds[0]:
#         cv2.rectangle(img,(w[0], w[1]),(w[2], w[3]),(0,0,255),1)
#     cv2.imwrite('/home2/sreevatsa/output_test_doctrv2_{}_{}.png'.format(d,formatted_time), img) 

def visualized_rescaled_bboxes_from_cropped(img_cropped,img_source,top_left):
    # rescaled_bboxes_from_cropped(img_cropped,img_source)
    preds = rescaled_bboxes_from_cropped(img_cropped,img_source,top_left)
    img = cv2.cvtColor(cv2.imread(img_source),cv2.COLOR_BGR2RGB)
    for w in preds[0]:
        cv2.rectangle(img,(w[0], w[1]),(w[2], w[3]),(0,0,255),1)
    # plt.imshow(img)
    cv2.imwrite('/home2/sreevatsa/afterfixoutput_test_doctrv2_{}_{}.png'.format(d,formatted_time), cv2.cvtColor(img,cv2.COLOR_RGB2BGR))

def save_cropped(img_dir):
    img = cv2.cvtColor(cv2.imread(img_dir),cv2.COLOR_BGR2RGB)
    org_TL = (0,0)
    org_BR = (img.shape[1],img.shape[0])

    preds = doctr_predictions_dir(img_dir)
    # print(preds)

    top1=[]
    left1=[]
    bottom1=[]
    right1 = []

    for i in preds[0]:
        left1.append(i[0])
        top1.append(i[1])
        right1.append(i[2])
        bottom1.append(i[3])

    # print(type(left1),type(right1))

    l = min(left1)
    r = max(right1)
    t = min(top1)
    b = max(bottom1)
    # print(l,r,t,b)

    top_left = (l-20,t-20)
    bottom_right = (r+20,b+20)

    print('cropped')
    print(top_left, bottom_right)

    difference_TL = (top_left[0]-org_TL[0],top_left[1]-org_TL[1])
    difference_BR = (abs(bottom_right[0]-org_BR[0]),abs(bottom_right[1]-org_BR[1]))
    print("ZXCV")
    print(difference_TL,difference_BR)

    # print(top_left, bottom_right)  
    x1,y1 = top_left
    x2,y2 = bottom_right

    cv2.rectangle(img,top_left, bottom_right,(0,255,255),2)
    # plt.imshow(img1)
    imgg1 = img[y1:y2, x1:x2]
    # plt.imshow(imgg1)
    cv2.imwrite('/home2/sreevatsa/cropped.png',imgg1)

    # print('JKGHF')
    # print(type(imgg1))
    # visualize_preds_dir('/home2/sreevatsa/a1.png')
    print(imgg1.shape)
    return top_left,imgg1

# preds = doctr_predictions_dir(args.ImageFile)
# print(preds)

# top1=[]
# left1=[]
# bottom1=[]
# right1 = []

# for i in preds[0]:
#     left1.append(i[0])
#     top1.append(i[1])
#     right1.append(i[2])
#     bottom1.append(i[3])

# print(type(left1),type(right1))

# l = min(left1)
# r = max(right1)
# t = min(top1)
# b = max(bottom1)
# print(l,r,t,b)

# top_left = (l-20,t-20)
# bottom_right = (r+20,b+20)

# print(top_left, bottom_right)  
# x1,y1 = top_left
# x2,y2 = bottom_right

# cv2.rectangle(img1,top_left, bottom_right,(0,255,255),2)
# # plt.imshow(img1)
# imgg1 = img1[y1:y2, x1:x2]
# # plt.imshow(imgg1)
# cv2.imwrite('/home2/sreevatsa/a1.png',imgg1)

# print('JKGHF')
# print(type(imgg1))



visualize_preds_dir(args.ImageFile)
cropped_TL, img_cropped = save_cropped(args.ImageFile)
# visualize_preds_dir('/home2/sreevatsa/cropped.png')
visualized_rescaled_bboxes_from_cropped(img_cropped,args.ImageFile,cropped_TL)