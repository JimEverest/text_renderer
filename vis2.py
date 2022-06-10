import argparse
from glob import glob
from pathlib import Path
import cv2
import numpy
import matplotlib.pyplot as plt
import os
import random
import json
import argparse

font = cv2.FONT_HERSHEY_SIMPLEX
img_path="example_data/output/aaa/images/"
labelName = "labels" # NO EXT (ie. txt/json)

#Note1: system will find label.txt under parent folder (root=-1)

def readLabel_TXT_byName(Name, filePath="PaddleOCR/rec_data_lesson_demo/train.txt",):
    # return the 1st Name occurrence line. 
    msg=""
    with open(filePath) as fh:
        for line in fh:
            if Name in line:
                msg=line
                break
    split_string = msg.split("\t")
    if(len(split_string)>1):
        #split a line on the TAB 
        return split_string[0],split_string[1]
    else:
        #split a line on the 1st space occurrence
        split_string = msg.split(" ",1)
        return split_string[0],split_string[1]


def readLabel_TXT_byID(filePath="PaddleOCR/rec_data_lesson_demo/train.txt",lineID=0):
    fp = open(filePath)
    msg=""
    for i, line in enumerate(fp):
        if i == lineID:
            msg=line
            break
    fp.close()
    split_string = msg.split("\t")

    if(len(split_string)>1):
        #split a line on the TAB 
        return split_string[0],split_string[1]
    else:
        #split a line on the 1st space occurrence
        split_string = msg.split(" ",1)
        return split_string[0],split_string[1]



def loadImg(fName = "",base=""):
  fp=os.path.join(base,fName)
#   print(fp)
  img = cv2.imread(fp)
  if img is None:
        result = "Image is empty!!"
        print("None Image, path not found-------------------->", fp)
  return img


def getLineNum(lbl): 
  file = open(lbl, "r")
  line_count = 0
  for line in file:
      if line != "\n":
          line_count += 1
  file.close()
  return(line_count)


def text(img, js,idx=0):
  txt = js
  tup_pt=(0,10)
  cv2.putText(img, txt, tup_pt, font, 0.5, (30, 30, 255), 1, cv2.LINE_AA)
  os.makedirs("./temp", exist_ok=True)
  cv2.imwrite('./temp/temp'+str(idx)+'.jpg', img)
  return img


def get_lbl_from_parent_lbl(img_name, root=0):
    # img_name ---> "000001.jpg"
    # img_path(global) --> */images/ ---> */labels.json

    root_path=Path(img_path)
    if(root==0):
        root_path=str(root_path)
    if(root==-1):
        root_path=str(root_path.parent)
    if(root==-2):
        root_path=str(root_path.parent.parent)
    if(root==-3):
        root_path=str(root_path.parent.parent.parent)
    
    # 1. try to find json
    lbl_path = os.path.join(root_path,labelName+".json")
    if os.path.isfile(lbl_path):
        with open(lbl_path, 'r') as f:
            data = json.load(f)
            img_id = img_name.split(".")[0]
            # print(data["labels"][img_id]
            return data["labels"][img_id]
    else:
        # 2. try to find label.txt under root path
        lbl_path = os.path.join(root_path,labelName+".txt")
        _img_pth,gt = readLabel_TXT_byName(Name=img_name,filePath=lbl_path)
        return gt





#region sample methods:
def random_sample_GT(lbl="",num=1,img_base="images/"):
    samples=num
    linC=getLineNum(lbl)
    print("Total lines: ",linC)
    rl = list(range(0,linC))
    rrl = (random.sample(rl, samples))
    imgs=[]
    gts=[]
    for i in rrl:
        fName,gt = readLabel_TXT_byID(filePath=lbl,lineID=i)
        # print(fName,i,gt)
        img = loadImg(fName,base=img_base)
        # img = text(img,gt)
        imgs.append(img)
        gts.append(gt)
    return imgs,gts

def random_sample_folder(lbl,num=1,gt=True,lbl_root=0):
    global img_path
    img_path = lbl
    samples=num
    filenames= os.listdir(lbl)
    random.shuffle(filenames)
    list_for_fNames=filenames[:samples]
    # targets=[]
    imgs=[]
    gts=[]
    for item in list_for_fNames:
        # item ---> "000001.jpg"
        _gt=None
        if gt:
            _gt=get_lbl_from_parent_lbl(item,root=lbl_root)
        # print("lbl--->", _gt)
        gts.append(_gt)
        # targets.append(item)
        img = loadImg(item,base=lbl)
        imgs.append(img)
    return imgs,gts
#endregion


#region  、、、、Show
def showImgs_code(imgs,lbls=None, r=6,c=4, savefile_name = "zamples"):
    figure = plt.figure(figsize = (c*3,r*1))
    for i in range(1,c*r+1):
        # img_pth = random.choice(os.listdir(folder))
        # label = img_pth.split("_")[1].split(".jpg")[0]
        # img = cv2.imread(os.path.join(folder,img_pth),0)
        
        figure.add_subplot(r,c,i)
        if lbls is not None:
            plt.title(lbls[i-1])
        plt.axis("off")
        plt.imshow(cv2.cvtColor(imgs[i-1], cv2.COLOR_BGR2RGB))
        # print(label)
    plt.savefig(savefile_name+".png", dpi=100)
    os.system("code "+savefile_name+".png")
    # plt.show()



def showImg(img):
    if(isinstance(img, str)):
        if os.path.isfile(img):
            loadImg(fName=img)
        if os.path.isdir(img):
            # random select
            pass
    img = loadImg(img)
    # figure = plt.figure(figsize = (cols*3,rows*1))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

#endregion


#1. folder
#2. file
#3. np array




if __name__ =="__main__":
    img_pth="example_data/output/chars_2000/images"
    imgs,lbls = random_sample_folder(lbl=img_pth,num=24, lbl_root=-1)# 
    # imgs,lbls = random_sample_folder(lbl="example_data/output/bank_chars_1/images")
    fn=img_pth.split("/")[-2]
    showImgs_code(imgs,lbls,savefile_name=fn)

  # imgs,gts = random_sample_GT(lbl="/home/jim/AI/ocr1/labels1.txt")
  # showImgs_code(imgs,gts,r=1,c=1)



# Usage:

# i,x = random_sample_folder("/home/jim/AI/ocr1/temp",num=2,lbl_root=-1)
# print (i[0].shape)
# print (x[0])


