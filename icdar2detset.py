# 0 -10000
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import tqdm 

#生成文字检测数据集
# seed=42时，生成了 train 7156 | val 776

np.random.seed(42)
train_cnt = 0
val_cnt = 0
for i in tqdm.tqdm(range(1,10000+1)):
    root = '/workspace/str/icdar/train/'
    num = str(i)
    while len(num) < 5:
        num = '0' +num 
    img_path = root +  'img/tr_img_' + num + '.jpg'
    gt_path = root + 'gt/tr_img_' + num + '.txt'

    try:
        img_file = open(img_path)
    except:
        continue
    try:
        gt_file = open(gt_path)
    except:
        continue

    if np.random.rand() < 0.9:
        train_cnt += 1
        save_root = '/workspace/str/detset/train/'
        save_img_path = save_root + 'img/img_' + str(train_cnt) + '.jpg'
        
        try:
            img = Image.open(img_path)
        except:
            continue
        img.save(save_img_path)

        save_gt_path = save_root + 'gt/gt_img_' + str(train_cnt) + '.txt'
        save_gt_file = open(save_gt_path,'w')

        new_lines = []
        for line in gt_file.readlines():
            line_lst = line.split(',')
            if line_lst[9] =='###\n':
                continue
            line_lst.pop(8)
            new_line = ','.join(line_lst)
            new_lines.append(new_line)
        
        save_gt_file.writelines(new_lines)
        save_gt_file.close()
    else:
        val_cnt += 1
        save_root = '/workspace/str/detset/val/'
        save_img_path = save_root + 'img/img_' + str(val_cnt) + '.jpg'
        
        try:
            img = Image.open(img_path)
        except:
            continue
        img.save(save_img_path)

        save_gt_path = save_root + 'gt/gt_img_' + str(val_cnt) + '.txt'
        save_gt_file = open(save_gt_path,'w')

        new_lines = []
        for line in gt_file.readlines():
            line_lst = line.split(',')
            if line_lst[9] =='###\n':
                continue
            line_lst.pop(8)
            new_line = ','.join(line_lst)
            new_lines.append(new_line)
        
        save_gt_file.writelines(new_lines)
        save_gt_file.close()
    
    img_file.close()
    gt_file.close()

print('train',train_cnt)
print('val',val_cnt)


    
