import tqdm
import numpy as np
from shapely.geometry import Polygon

TP = 0
TOT_PRED = 0
TOT_GT = 0

def measure_TP(pred_bbox, gt_bbox, pred_transcript, gt_transcript):

    pred_bbox = np.array(pred_bbox)
    gt_bbox = np.array(gt_bbox)

    pred_bbox = pred_bbox.reshape([-1, 4, 2])
    gt_bbox = gt_bbox.reshape([-1, 4, 2])
    global TP 
    global TOT_GT
    global TOT_PRED
    
    TOT_GT += len(gt_bbox)
    TOT_PRED += len(pred_bbox)

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

for i in tqdm.tqdm(range(1,776+1)):
    gt_file_path = '/workspace/str/detset/annotations/test/gt_img_' + str(i) + '.txt'
    pred_file_path = '/workspace/str/myicdar/traingt/' +  str(i) + '.txt'

    try:
        gt_file = open(gt_file_path)
        pred_file = open(pred_file_path)
    except:
        print(gt_file_path)
        print(pred_file_path)
    
    pred_bbox = []
    gt_bbox = []
    pred_transcript = []
    gt_transcript = []

    for line in gt_file.readlines():
        line_lst = line.split(',')
        if line_lst[-1] =='###\n':
            continue
        gt_bbox.append(line_lst[:8])
        gt_transcript.append(line_lst[-1])

    for line in pred_file.readlines():
        line_lst = line.split(',')
        if line_lst[9] =='###\n':
            continue
        pred_bbox.append(line_lst[:8])
        pred_transcript.append(line_lst[-1])
    
    measure_TP(pred_bbox,gt_bbox,pred_transcript,gt_transcript)

precision = TP / TOT_PRED
recall = TP / TOT_GT
f1_score = 2 * precision * recall / (precision + recall + 1e-16)

print('precision',precision)
print('recall',recall)
print('f1_score',f1_score)