# The new config inherits a base config to highlight the necessary modification

_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

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
        bbox_head=dict(num_classes=1),
        ))

# runner = dict(type='EpochBasedRunner', max_epochs=24)
max_epochs = 1000
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# 0.0025
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(
     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[int(max_epochs*0.6), int(max_epochs*0.8)])

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ("car",)
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
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
        img_prefix='data/raw_dataset/RAW_converted',
        classes=classes,
        ann_file='data/raw_dataset/RAW_converted/annotations/train_ann.json',
        pipeline=train_pipeline),

    val=dict(
        img_prefix='data/raw_dataset/RAW_converted',
        classes=classes,
        ann_file='data/raw_dataset/RAW_converted/annotations/val_ann.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='data/raw_dataset/RAW_converted',
        classes=classes,
        ann_file='data/raw_dataset/RAW_converted/annotations/val_ann.json',
        pipeline=test_pipeline))

log_config = {'interval' :2, 
            'hooks': [dict(type='TensorboardLoggerHook'), dict(type='TextLoggerHook')]}

workflow = [('train', 1)]
work_dir = 'work_dirs/raw_data'
checkpoint_config = dict(interval=1, save_optimizer=True,max_keep_ckpts=10)
# load_from = 'work_dirs/stride_4/maxpool_near/lr02_b2_true/finetune/epoch_9.pth'
# resume_from = 'work_dirs/normal_data/quantized/s6_k7_lr02/finetune/epoch_14.pth'
load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
# We can use the pre-trained Mask RCNN model to obtain higher performance
# https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth