# The new config inherits a base config to highlight the necessary modification


_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
#_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        ))

runner = dict(type='EpochBasedRunner', max_epochs=24)


optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='RMSprop',lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0001, momentum=0.9, centered=False)
#optimizer = dict(_delete_=True, type='ASGD', lr=0.001, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='LBFGS', lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)


optimizer_config = dict(
     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ("person", "rider", "car", "bus", "truck", "bike",
               "motor", "traffic light", "traffic sign", "train")
data = dict(
    train=dict(
        img_prefix='data/bdd100k/train/',
        classes=classes,
        ann_file='data/bdd100k/annotations/det_train_coco.json'),
    val=dict(
        img_prefix='data/bdd100k/val/',
        classes=classes,
        ann_file='data/bdd100k/annotations/det_val_coco.json'),
    test=dict(
        img_prefix='data/bdd100k/val/',
        classes=classes,
        ann_file='data/bdd100k/annotations/det_val_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'nas/home/shunlin/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'