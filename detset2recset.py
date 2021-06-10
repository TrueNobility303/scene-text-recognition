import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import tqdm 
from mmocr.datasets.pipelines.crop import crop_img
import mmcv

np.random.seed(42)
train_cnt = 0
test_cnt = 0

#detset test:1 to 776, train:1 to 7156


#从detset生成recset用于ocr任务
# recset概览： train 63520 | test 6321

#生成test数据集

NUM_DET_TRAIN = 7156 
NUM_DET_TEST = 776

save_train_labels_file = open('/workspace/str/recset/train_label.txt','w')
save_test_labels_file = open('/workspace/str/recset/test_label.txt','w') 

train_label_lines = []
test_label_lines = []

for i in tqdm.tqdm(range(1,NUM_DET_TEST)):
    img_path = '/workspace/str/detset/imgs/test/img_' +  str(i) + '.jpg'
    gt_path = '/workspace/str/detset/annotations/test/gt_img_' +  str(i) + '.txt'
    
    try:
        gt_file = open(gt_path)
    except:
        continue    
    try:
        img = mmcv.imread(img_path)
    except:
        print('cannot open img_file', img_path)
        continue

    for line in gt_file.readlines():
        bbox = line.split(',')
        #print(bbox)
        script = bbox[8]
        bbox.pop(8)
        if len(bbox) != 8:
            continue
        bbox = [int(_x) for _x in bbox]
        #print(bbox)
        box_img = crop_img(img, bbox)

        test_cnt += 1
        relevant_path = 'test_img/word_' + str(test_cnt) + '.png'
        save_img_path = '/workspace/str/recset/' + relevant_path 

        try:
            mmcv.imwrite(box_img,save_img_path)
        except:
            test_cnt -= 1 

        test_label_lines.append(relevant_path + ' ' + script)

    gt_file.close()

save_test_labels_file.writelines(test_label_lines)

#生成train数据集
for i in tqdm.tqdm(range(1,NUM_DET_TRAIN)):
    img_path = '/workspace/str/detset/imgs/training/img_' +  str(i) + '.jpg'
    gt_path = '/workspace/str/detset/annotations/training/gt_img_' +  str(i) + '.txt'
    
    try:
        gt_file = open(gt_path)
    except:
        continue    
    try:
        img = mmcv.imread(img_path)
    except:
        print('cannot open img_file', img_path)
        continue

    for line in gt_file.readlines():
        bbox = line.split(',')
        #print(bbox)
        script = bbox[8]
        bbox.pop(8)
        if len(bbox) != 8:
            continue
        bbox = [int(_x) for _x in bbox]
        #print(bbox)
        box_img = crop_img(img, bbox)

        train_cnt += 1
        relevant_path = 'train_img/word_' + str(train_cnt) + '.png'
        save_img_path = '/workspace/str/recset/' + relevant_path 

        try:
            mmcv.imwrite(box_img,save_img_path)
        except:
            train_cnt -= 1

        train_label_lines.append(relevant_path + ' ' + script)
    gt_file.close()

save_train_labels_file.writelines(train_label_lines)

print('train',train_cnt)
print('test',test_cnt)