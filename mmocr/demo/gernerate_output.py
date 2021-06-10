from argparse import ArgumentParser

import mmcv
import torch
from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.pipelines.crop import crop_img
import tqdm
import numpy as np
from shapely.geometry import Polygon
import os 

def measure_TP(pred_bbox, gt_bbox, pred_transcript, gt_transcript):
    pred_bbox = pred_bbox.reshape([-1, 4, 2])
    gt_bbox = gt_bbox.reshape([-1, 4, 2])

    TP = 0

    for gt_b, gt_t in zip(gt_bbox, gt_transcript):
        gt_quad = Polygon(gt_b).convex_hull
        gt_area = gt_quad.area

        for pred, pred_trans in zip(pred_bbox, pred_transcript):
            pred_quad = Polygon(pred).convex_hull

            if gt_quad.intersects(pred_quad):
                inter = gt_quad.intersection(pred_quad).area
                iou = inter / (gt_area + pred_quad.area - inter)

                if iou > 0.5:
                    if gt_t==pred_trans or gt_t=="###" or gt_t=="":
                        TP += 1
                        break
    
    #precision = TP / pred_total
    #recall = TP / gt_total
    #f1_score = 2 * precision * recall / (precision + recall + 1e-16)
    
    return TP

#生成预测结果在指定路径内
def det_and_recog_f1_measure(det_model, recog_model, image_path, outfile_path):
    
    new_lines = []

    end2end_res = {'filename': image_path}
    end2end_res['result'] = []

    image = mmcv.imread(image_path)
    det_result = model_inference(det_model, image)
    bboxes = det_result['boundary_result']

    box_imgs = []
    for bbox in bboxes:
        box_res = {}
        box_res['box'] = [round(x) for x in bbox[:-1]]
        box_res['box_score'] = float(bbox[-1])
        box = bbox[:8]
        if len(bbox) > 9:
            min_x = min(bbox[0:-1:2])
            min_y = min(bbox[1:-1:2])
            max_x = max(bbox[0:-1:2])
            max_y = max(bbox[1:-1:2])
            box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        box_img = crop_img(image, box)

        #not use batch mode for ocr/text reconition

        recog_result = model_inference(recog_model, box_img)
        text = recog_result['text']
        text_score = recog_result['score']
        if isinstance(text_score, list):
            text_score = sum(text_score) / max(1, len(text))
        box_res['text'] = text
        box_res['text_score'] = text_score
        
        new_a_line = ''
        for i in range(8):
            new_a_line += str(round(bbox[i]))
            new_a_line += ','
        new_a_line += 'Latin,'
        new_a_line += text
        new_a_line += '\n'
        #print(new_a_line)
        new_lines.append(new_a_line)
        end2end_res['result'].append(box_res)

    try:
        outfile = open(outfile_path,'w')
        outfile.writelines(new_lines)
        outfile.close()
    except:
        print(outfile_path)
        pass 

    return end2end_res

def main():
    #指定模型，config、checkpoints等
    device = 'cpu'
    det_config = '/workspace/str/psenet_160/my_psenet.py'
    det_ckpt = '/workspace/str/psenet_160/latest.pth'

    recog_config = '/workspace/str/rbsan_200/my_robust_scanner.py'
    recog_ckpt = '/workspace/str/rbsan_200/latest.pth'

    detect_model = init_detector(
        det_config, det_ckpt, device=device)
    if hasattr(detect_model, 'module'):
        detect_model = detect_model.module
    if detect_model.cfg.data.test['type'] == 'ConcatDataset':
        detect_model.cfg.data.test.pipeline = \
            detect_model.cfg.data.test['datasets'][0].pipeline

    # build recog model
    recog_model = init_detector(
        recog_config, recog_ckpt, device=device)
    if hasattr(recog_model, 'module'):
        recog_model = recog_model.module
    if recog_model.cfg.data.test['type'] == 'ConcatDataset':
        recog_model.cfg.data.test.pipeline = \
            recog_model.cfg.data.test['datasets'][0].pipeline
    
    #生成结果并且输出至myicdar文件夹下
    for i in tqdm.tqdm(range(80,9999+1)):

        #for val
        #img_path = '/workspace/str/detset/imgs/test/img_' + str(i) + '.jpg'
        #output_path = '/workspace/str/myicdar/traingt/' +  str(i) + '.txt'
        
        num = str(i)
        while len(num) < 5:
            num = '0' + num 
        
        img_path = '/workspace/str/icdar/test/img/tr_img_' + num + '.jpg'
        output_path = '/workspace/str/myicdar/testgt/' +  num + '.txt'
        if not os.path.isfile(img_path):
            #print(img_path)
            continue

        det_recog_result = det_and_recog_f1_measure(detect_model, recog_model,img_path,output_path)

if __name__ == '__main__':
    main()
