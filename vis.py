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


def readLabel(fp="PaddleOCR/rec_data_lesson_demo/train.txt",id=0):
  fp = open(fp)
  msg=""
  for i, line in enumerate(fp):
      if i == id:
          msg=line
          break
  fp.close()
  split_string = msg.split("\t")
  return split_string[0],split_string[1]

def loadImg(fName = "",base="/content/PaddleOCR/train_data/icdar2015/text_localization/"):
  fp=os.path.join(base,fName)
  img = cv2.imread(fp)
  return img


def getLineNum(lbl):
  file = open(lbl, "r")
  line_count = 0
  for line in file:
      if line != "\n":
          line_count += 1
  file.close()
  return(line_count)


def ploy(img, js,idx=0):
  
  txt = js
  tup_pt=(0,10)
  cv2.putText(img, txt, tup_pt, font, 0.5, (30, 30, 255), 1, cv2.LINE_AA)

  os.makedirs("../temp", exist_ok=True)
  cv2.imwrite('../temp/temp'+str(idx)+'.jpg', img)
  return img

def get_lbl_ImgName(img_name):
    # Path(img_name).parent()
    root_path=str(Path(img_path).parent)
    # root_path=img_path
    json_path = os.path.join(root_path,"labels.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
        img_id = img_name.split(".")[0]
        # print(data["labels"][img_id]
        
        return data["labels"][img_id]


def random_sample_folder(lbl,r=4,c=6,addLbl=False):
    global img_path
    img_path = lbl
    samples=r*c
    filenames= os.listdir(lbl)
    random.shuffle(filenames)
    list_for_centers=filenames[:samples]
    print(list_for_centers)
    targets=[]
    imgs=[]
    lbls=[]
    for item in list_for_centers:
        # target=lbl+item
        if addLbl:
          txt=get_lbl_ImgName(item)
          # print("lbl--->", txt)
          lbls.append(txt)
          targets.append(item)
        img = loadImg(item,base=lbl)
        imgs.append(img)
    return imgs,lbls


# def random_sample(lbl="/content/PaddleOCR/train_data/icdar2015/text_localization/train_icdar2015_label.txt",r=4,c=6,img_base="/content/PaddleOCR/train_data/icdar2015/text_localization/",addLbl=False):
#   samples=r*c
#   linC=getLineNum(lbl)
#   # print(linC)
#   rl = list(range(0,linC))
#   rrl = (random.sample(rl, samples))
#   imgs=[]
#   for i in rrl:
#     xxx,js = readLabel(fp=lbl,id=i)
#     # print(i,js)
#     img = loadImg(xxx,base=img_base)
#     if(addLbl):
#       img = ploy(img,js)
#     imgs.append(img)
#   return imgs

def showImgs(imgs,lbls=None, rows=6,cols=4, savefile_name = "zamples"):
    figure = plt.figure(figsize = (cols*3,rows*1))
    for i in range(1,cols*rows+1):
        # img_pth = random.choice(os.listdir(folder))
        # label = img_pth.split("_")[1].split(".jpg")[0]
        # img = cv2.imread(os.path.join(folder,img_pth),0)
        
        figure.add_subplot(rows,cols,i)
        try:
          plt.title(lbls[i-1])
        except:
          pass
        plt.axis("off")
        plt.imshow(cv2.cvtColor(imgs[i-1], cv2.COLOR_BGR2RGB))
        # print(label)
    plt.savefig(savefile_name+".png", dpi=100)
    os.system("code "+savefile_name+".png")
    # plt.show()

def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument("--look",default="example_data/output/bank_chars_1")
  return parser.parse_args()


# xxx,js = readLabel(id=0)
# img = loadImg(xxx)
# ploy(img,js)



if __name__=="__main__":
  # --look example_data/output/chars_2000/images
  args = parse_arg()
  imgs,lbls = random_sample_folder(lbl=args.look)
  # imgs,lbls = random_sample_folder(lbl="example_data/output/bank_chars_1/images")
  fn=args.look.split("/")[-2]
  showImgs(imgs,lbls,savefile_name=fn)



# =============== Hard coded Folder
# "/mnt/e/ai_data/Chinese_dataset/images"
# "/mnt/e/ai_data/Synthetic Chinese String Dataset/images"

# if __name__=="__main__":
#   img_folder = "/mnt/e/ai_data/Synthetic Chinese String Dataset/images"
#   imgs,lbls = random_sample_folder(lbl=img_folder)
#   # imgs,lbls = random_sample_folder(lbl="example_data/output/bank_chars_1/images")
#   fn=img_folder.split("/")[-2]
#   showImgs(imgs,lbls,savefile_name=fn)



