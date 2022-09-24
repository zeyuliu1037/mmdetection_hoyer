_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe'),
            # loss_conv=dict(type='L1Loss', loss_weight=0.0),
            ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=10),
        ))

dataset_type = 'COCODataset'
classes = ("person", "rider", "car", "bus", "truck", "bike",
               "motor", "traffic light", "traffic sign", "train")
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)

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
        img_prefix='data/bdd100k/train/',
        classes=classes,
        ann_file='data/bdd100k/annotations/det_train_coco.json',
        pipeline=train_pipeline),

    val=dict(
        img_prefix='data/bdd100k/val/',
        classes=classes,
        ann_file='data/bdd100k/annotations/det_val_coco.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='data/bdd100k/val/',
        classes=classes,
        ann_file='data/bdd100k/annotations/det_val_coco.json',
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 15])
runner = dict(type='EpochBasedRunner', max_epochs=18)
log_config = dict(
    interval=50,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])

resume_from = 'work_dirs/normal_data/bn_true/0025/finetune/latest.pth'
workflow = [('train', 1)]

work_dir = 'work_dirs/normal_data/bn_true/0025/finetune'

