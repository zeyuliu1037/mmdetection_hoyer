import json
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)
# from mmcv import Config, DictAction
import numpy as np
from PIL import Image
# import scipy
# from mmdet.datasets.coco import CocoDataset
# from functools import partial
# from mmcv.parallel import collate
# from torch.utils.data import DataLoader
import os
import math

# gt_json = 'data/raw_dataset/RAW_converted/annotations/val_ann.json'
# pred_json = 'work_dirs/yolof/bdd_test/result.bbox.json'
# pred_json = 'work_dirs/yolof/bdd_test/pretrain_result.bbox.json'

# gt_json = 'data/bdd100k/annotations/det_val_coco.json'
# pred_json = 'work_dirs/yolof/bdd_test/bdd2/result.bbox.json'

# gt_json = 'data/raw_dataset/RAW_converted/annotations/test_all_ann.json'
# pred_json = 'work_dirs/coco_raw/frcnn_lr_0.002_use_mixch/bbox.json'
# pred num: 10904, right num: 368, gt num: 394, acc : 0.934010152284264
# pred_json = 'work_dirs/coco_raw/frcnn_lr_0.002_use_mixch/pretrained.bbox.json'
# pred num: 5509, right num: 368, gt num: 394, acc : 0.934010152284264
gt_json = 'data/raw_dataset/RAW_converted/annotations/test_all_annX2.json'
pred_json = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.bbox.json' 
# pred num: 5679, right num: 369, gt num: 394, acc : 0.9365482233502538
# pred_json = 'work_dirs/coco_raw/frcnn_lr_0.002_frozen/epoch_12.bbox.json'
# pred num: 8874, right num: 369, gt num: 394, acc : 0.9365482233502538
# pred_json = 'work_dirs/coco_raw/frcnn_lr_0.01_after_demosaic_new/bbox.json' 
# pred num: 4752, right num: 359, gt num: 394, acc : 0.9111675126903553


def iou(a,b):
    # [top_left_x, top_left_y, weight, height]
    s_a = a[2]*a[3]
    s_b = b[2]*b[3]

    w = min(b[0] + b[2], a[0] + a[2]) - max(a[0], b[0])
    h = min(b[1] + b[3], a[1] + a[3]) - max(a[1], b[1])

    if w <= 0 or h <=0:
        return 0
    s_c = w*h
    return s_c / (s_a + s_b - s_c)

def test():
    with open(pred_json) as f:
        pred_data = json.load(f)
    with open(gt_json) as f:
        gt_data = (json.load(f))['annotations']

    gt_dict = {}
    for gt in gt_data:
        id = gt['image_id']
        if id in gt_dict:
            gt_dict[id].append(gt)
        else:
            gt_dict[id] = [gt]
    counter = set()
    for item in pred_data:
        if item['score'] > 0.0:
            id = item['image_id']
            if id in gt_dict:
                for gt_bbox in gt_dict[id]:
                    #  and item['category_id'] == gt_bbox['category_id']
                    if iou(item['bbox'], gt_bbox['bbox']) > 0.6:
                        # print('the right sample id is {}, score is: {},pred bbox is {}, bbox: is {}'.format(id, item['score'], item['bbox'], gt_bbox))
                        counter.add(gt_bbox['id'])
    print('pred num: {}, right num: {}, gt num: {}, acc : {}'.format(len(pred_data), len(counter), len(gt_data), len(counter)/len(gt_data)))

def dataset_test():
    # cfg = Config.fromfile('work_dirs/raw_data/raw_image_config.py')
    # print(cfg.data.test)
    # coco = build_dataset(cfg.data.test)
    # print(coco)
    # fc = mmcv.FileClient(backend='disk')
    # r1 = fc.get('data/raw_dataset/RAW_converted/train/1.png')
    # r1 = mmcv.imfrombytes(r1, flag='color')
    # print(type(r1), len(r1))
    # r2 = np.load('/nas/home/gdatta/raw_image_contents/demosaiced_image/raw_pred_COCO_val2014_000000522418.npy')
    r2 = np.load('/nas/home/gdatta/raw_image_contents/demosaiced_image/raw_pred_COCO_val2014_000000391895.npy')
    r4 = scipy.ndimage.zoom(r2, [2,2,1])
    image = Image.fromarray((r2 * 255).astype(np.uint8))
    image2 = Image.fromarray((r4 * 255).astype(np.uint8))
    image.save('data/000000391895.jpg')
    image2.save('data/000000391895x2.jpg')
    print(type(r2), len(r2), r2.shape)
    print(type(r4), len(r4), r4.shape)
    # with open('/nas/home/gdatta/raw_image_contents/demosaiced_image/raw_pred_COCO_val2014_000000391895.npy', 'rb') as f:
    #     r3 = f.read() 
    #     r3 = mmcv.imfrombytes(r3, flag='color')
    #     print(type(r3), len(r3))
    # print(r1.shape)
    with open('data/ann/instances_val2014.json') as f:
    # with open('data/ann/instances_minival.json') as f:
        coco_json = json.load(f)
        print('val2014 {} images'.format(len(coco_json['images'])))
    with open('data/ann/instances_train2014.json') as f:
        train = json.load(f)
        print('train2014 {} images'.format(len(train['images'])))
    with open('data/ann/instances_maxitrain.json') as f:
        train = json.load(f)
        print('maxitrain {} images'.format(len(train['images'])))
    with open('data/ann/instances_minival.json') as f:
        train = json.load(f)
        print('minival {} images'.format(len(train['images'])))
    print(coco_json['annotations'][0].keys())
    img = coco_json['images'][0]
    print('image filename: {}, w: {}, h: {}, id: {}'.format(img['file_name'], img['width'], img['height'], img['id']))
    all_classes = {}
    for c in coco_json['categories']:
        all_classes[c['id']] = c['name']
    for ann in coco_json['annotations']:
        if ann['image_id'] == 391895:
            print('bbox: {}, category_id: {}, cat: {}'.format(ann['bbox'], ann['category_id'], all_classes[ann['category_id']]))
    # print('image id: {}, area: {}, bbox: {}'.format(coco_json['annotations'][0]['image_id'], coco_json['annotations'][0]['area'], coco_json['annotations'][0]['bbox']))
def dataset_check():
    with open('data/ann/instances_maxitrain.json') as f:
    # with open('data/ann/instances_train2014.json') as f:
        coco_json = json.load(f)
        print('maxitrain {} images'.format(len(coco_json['images'])))
    images = coco_json['images']
    w = []
    h = []
    for i,img in enumerate(images):
        w.append(img['width'])
        h.append(img['height'])
        if i == 59403 or i == 59894:
            name = img['file_name'][:-4]
            filename = f'/nas/home/gdatta/raw_image_contents/demosaiced_image/raw_pred_{name}.npy'
            npy_img = np.load(filename)
            image = Image.fromarray((npy_img * 255).astype(np.uint8))
            image.save(f'data/{name}.jpg')
            print('neme: {}, npy_img shape: {}, h: {}, W: {}, min: {}, max: {}, mean: {}'.format(name, npy_img.shape, img['width'], img['height'], np.min(npy_img), np.max(npy_img), np.mean(npy_img)))
    all_classes = {}
    for c in coco_json['categories']:
        all_classes[c['id']] = c['name']
    for ann in coco_json['annotations']:
        if ann['image_id'] == 363747 or ann['image_id'] == 187714:
            print(ann['image_id'])
            print('bbox: {}, category_id: {}, cat: {}'.format(ann['bbox'], ann['category_id'], all_classes[ann['category_id']]))

    w,h = np.array(w), np.array(h)
    print('w max: {}, min: {}, mean: {}; h max: {}, min: {}, mean: {}'.format(np.max(w), np.min(w), np.mean(w), np.max(h), np.min(h), np.mean(h)))
    print('argmin: {}, {}'.format(np.argmin(w), np.argmin(h))) # 59403, 59894




def coco_test():
    NumberCOCOImages = 8 # 决定batch_size有多大
    data_root = '/'# COCO数据集的路径
    imgs_per_gpu = NumberCOCOImages
    num_gpus = 1
    workers_per_gpu = 1
    batch_size = num_gpus * imgs_per_gpu


    num_workers = num_gpus * workers_per_gpu

    load_pipeline = [
        dict(type='LoadImageFromNpy'),
        dict(type='LoadImageFromNpy'),
        dict(type='Resize', img_scale=(224, 224), keep_ratio=False),#对原始图像进行缩放，全部变为224*224的分辨率
        dict(type='ImageToTensor', keys=['img']),#这个不能少，否则无法加载数据集
        dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor'))] #这个不能少，否则无法加载数据集
        # 除了上述这些pipeline，还可以根绝需要参考官方配置文件进行增减
        
    CocoDataset_rgb = CocoDataset('data/ann/instances_maxitrain.json', load_pipeline,
                                data_root='/',
                                img_prefix='nas/home/gdatta/raw_image_contents/demosaiced_image/',
                                test_mode=True
                                )
                                
    data_loader_rgb = DataLoader(
        CocoDataset_rgb,
        batch_size=batch_size,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        num_workers=num_workers)
        
    for i_rgb, data_rgb in enumerate(data_loader_rgb):
        img = data_rgb['img']
        print(img.shape)
        break


def demosaic(raw):
    H, W = raw.shape
    if H%2 == 1:
        H = H - 1
    if W%2 == 1:
        W = W - 1
    de_raw = np.zeros((math.floor(H/2), math.floor(W/2), 3))

    de_raw[0:H:1, 0:W:1, 0] = raw[0:H:2, 0:W:2]
    de_raw[0:H:1, 0:W:1, 2] = raw[1:H:2, 1:W:2]
    de_raw[0:H:1, 0:W:1, 1] = (raw[0:H:2, 1:W:2] + raw[1:H:2, 0:W:2])/2

    return de_raw

def npy_check():
    with open('data/ann/instances_maxitrain.json') as f:
    # with open('data/ann/instances_train2014.json') as f:
        coco_json = json.load(f)
        print('maxitrain {} images'.format(len(coco_json['images'])))
    name = coco_json['images'][0]['file_name'][:-4]
    filename = f'/nas/home/gdatta/raw_image_contents/demosaiced_image/raw_pred_{name}.npy'
    coco_img = np.load(filename)
    # coco_img = (coco_img[:,:,0]*0.299 + coco_img[:,:,1]*0.587 + coco_img[:,:,2]*0.114)*255.0
    coco_img = coco_img[:,:,1]*255
    print('coco img: shape: {}, min: {}, max: {}'.format(coco_img.shape, np.min(coco_img), np.max(coco_img)))
    image = Image.fromarray((coco_img).astype(np.uint8))
    image.save(f'data/coco_2_ch_{name}.jpg')
    coco_img = demosaic(coco_img)
    print('coco img: shape: {}, min: {}, max: {}'.format(coco_img.shape, np.min(coco_img), np.max(coco_img)))
    image = Image.fromarray((coco_img).astype(np.uint8))
    image.save(f'data/coco_2_ch_demosaic{name}.jpg')
    # raw_image = np.load('data/raw_dataset/RAW_npy/test/1.npy')
    # raw_image = cv2.imread('data/raw_dataset/RAW_converted/val/191.png', cv2.IMREAD_GRAYSCALE)  
    # print('raw img: shape: {}, min: {}, max: {}'.format(raw_image.shape, np.min(raw_image), np.max(raw_image)))
    # # image = Image.fromarray((raw_image).astype(np.uint8))
    # # image.save(f'data/raw_1.jpg')
    # cv2.imwrite('data/raw_191.jpg', raw_image)
    # image_three = cv2.merge([raw_image,raw_image,raw_image])
    # raw_image = cv2.cvtColor(raw_image,cv2.COLOR_GRAY2RGB)
    # print(sum(image_three - raw_image))
    # print('raw img: shape: {}, min: {}, max: {}'.format(raw_image.shape, np.min(raw_image), np.max(raw_image)))
    # # image = Image.fromarray((raw_image).astype(np.uint8))
    # # image.save(f'data/raw_1_color.jpg')
    # cv2.imwrite('data/raw_191_color.jpg', raw_image)

def image_stat():
    files = os.listdir('data/raw_dataset/RAW_converted/train')
    means = []
    for file in files:
        file_array = Image.open(f'data/raw_dataset/RAW_converted/train/{file}')
        file_array = np.asarray(file_array)
        file_array = demosaic(file_array)
        # means.append([np.mean(file_array), np.min(file_array), np.max(file_array), np.std(file_array)])
        means.append([np.mean(file_array, (0,1)), np.min(file_array), np.max(file_array), np.std(file_array, (0,1))])
    means = np.asarray(means)
    print(means.shape)
    print(np.mean(means, 0))

    files = os.listdir('/nas/home/gdatta/raw_image_contents/demosaiced_image')
    demosaiced_means = []
    for i,file in enumerate(files):
        if i < 1000:
            file_array = np.load(f'/nas/home/gdatta/raw_image_contents/demosaiced_image/{file}')
            file_array = np.asarray(file_array)*255.0
            demosaiced_means.append([np.mean(file_array, (0,1)), np.min(file_array), np.max(file_array), np.std(file_array, (0,1))])
    demosaiced_means = np.asarray(demosaiced_means)
    print(demosaiced_means.shape)
    print(np.mean(demosaiced_means, 0))

def chech_folder():
    files = os.listdir('/nas/home/gdatta/raw_image_contents/demosaiced_image')
    for file in files:
        if file[-4:] != '.npy':   
            print(file)
    print(len(files))
 
if __name__ == '__main__':
    test()
    # dataset_test()
    # dataset_check()
    # npy_check()
    # image_stat()
    # chech_folder()
    # coco_test()
# bdd100:(true): 56.24 (score>0.5, iou>0.5) (104628/186033), pred num: 990635; 75.65 (score>0.1, iou>0.5) (140749/186033)
# bdd100: 48.46 (score>0.5, iou>0.5) (90158/186033), pred num: 990635; 66.45 (score>0.1, iou>0.5) (123625/186033)

# raw:    51.28 (score>0.0, iou>0.5) (20/39) pred num: 197; 46.15 (score>0.1, iou>0.5) (18/39)
# raw:    33.33 (score>0.0, iou>0.5) (13/39) pred num: 3500; 28.20 (score>0.1, iou>0.5) (11/39)

# after train: pred num: 7275, right num: 83, gt num: 394, acc : 0.21065989847715735
# before train: pred num: 5680, right num: 87, gt num: 394, acc : 0.22081218274111675