import json

f_vg = open("data/train_dataset.json", "r")
f_coco = open("data/obj_detection_coco_0.1.json", "r")
f_vgg_coco = open("data/obj_detection_vgg_coco_reg_0.03thre.json", "r")

vg_det = json.load(f_vg)
coco_det = json.load(f_coco)
vgg_coco_det = json.load(f_vgg_coco)

vg_proposals=0.
coco_proposals=0.
pascal_proposals = 0.
count =0.
for key, dic in vg_det.items():
    count+=1
    vg_proposals+=len(vg_det[key]["classes"])
    coco_proposals+=len(coco_det[key]["classes"])
    pascal_proposals+=len(vgg_coco_det[key]["classes"])
    if count>1000:
        break

print("vg avg proposals: ", vg_proposals/count)
print("coco avg proposals: ", coco_proposals/count)
print("vgg coco avg proposals: ", pascal_proposals/count)
