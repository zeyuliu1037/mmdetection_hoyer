# The new config inherits a base config to highlight the necessary modification


_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe'),
        # loss_conv=dict(type='L1Loss', loss_weight=0.0),
        ),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        ))

runner = dict(type='EpochBasedRunner', max_epochs=16)

# 0.0025
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

log_config = {'interval' :50, 
            'hooks': [
                dict(type='TensorboardLoggerHook'), 
                dict(type='TextLoggerHook')
            ]}

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 12])

optimizer_config = dict(
     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ("person", "rider", "car", "bus", "truck", "bike",
               "motor", "traffic light", "traffic sign", "train")

img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                           (1333, 768), (1333, 800)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize',**img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize',**img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
]

data = dict(
    samples_per_gpu=2,
    train=dict(
        img_prefix='/nas/home/zeyul/mmdetection_old/data/bdd100k/train/',
        classes=classes,
        ann_file='/nas/home/zeyul/mmdetection_old/data/bdd100k/annotations/det_train_coco.json',
        pipeline=train_pipeline),

    val=dict(
        img_prefix='/nas/home/zeyul/mmdetection_old/data/bdd100k/val/',
        classes=classes,
        ann_file='/nas/home/zeyul/mmdetection_old/data/bdd100k/annotations/det_val_coco.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='/nas/home/zeyul/mmdetection_old/data/bdd100k/val/',
        classes=classes,
        ann_file='/nas/home/zeyul/mmdetection_old/data/bdd100k/annotations/det_val_coco.json',
        pipeline=test_pipeline))

workflow = [('train', 1)]
# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'work_dirs/stride_4/maxpool_near/lr0025_b2/epoch_8.pth'
# load_from = 'work_dirs/baseline/lr02_b2_bn_train/finetune/epoch_8.pth'
work_dir = 'work_dirs/baseline/lr02_b2_bn_train/finetune_again'