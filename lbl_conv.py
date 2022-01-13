import json
import os

look_folder = "/home/jim/AI/text_renderer/example_data/output/"
json_path = "/home/jim/AI/text_renderer/example_data/output/imgaug_emboss_example/labels.json"
txt_path = os.path.join(look_folder,"labels.text") 


# directory_contents = os.listdir(look_folder)
# print("--->",directory_contents)
# for item in directory_contents:
#     if os.path.isdir(item):
#         print(item)



for root,dirs,files in os.walk(look_folder):
    for dir in dirs:
        cur_pth= os.path.join(root,dir)
        if any(fname.endswith('.json') for fname in os.listdir(cur_pth)):
            print("--->",dir)
            print(cur_pth)
            json_path = os.path.join(cur_pth,"labels.json")

            with open(json_path, 'r') as f:
                data = json.load(f)

            with open(txt_path, 'a') as f:
                for k in data["labels"]:
                    # print(k)
                    # print(data["labels"][k])
                    _s = dir + "\\" +k + ".jpg	" + data["labels"][k]+"\n"
                    f.writelines(_s) 
































