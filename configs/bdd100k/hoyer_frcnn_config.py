# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='HoyerResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/resnet50_bn_spike_conv_0.8_SGD_best.pt')))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=False)
    # mean=[47.039, 35.467, 42.159], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        # img_scale=[(1333, 640), (1333, 800)],
        # multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    train=dict(
        pipeline=train_pipeline,
        img_prefix='/nas/home/gdatta/data/coco/train2017',
        ann_file='/nas/home/gdatta/data/coco/annotations/instances_train2017.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='/nas/home/gdatta/data/coco/val2017',
        ann_file='/nas/home/gdatta/data/coco/annotations/instances_val2017.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='/nas/home/gdatta/data/coco/val2017',
        ann_file='/nas/home/gdatta/data/coco/annotations/instances_val2017.json'))

# runner = dict(type='EpochBasedRunner', max_epochs=24)


# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='RMSprop',lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0001, momentum=0.9, centered=False)
#optimizer = dict(_delete_=True, type='ASGD', lr=0.001, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='LBFGS', lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
log_config = dict(interval=500, hooks=[dict(type='TextLoggerHook')])


# optimizer_config = dict(
#      _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# Modify dataset related settings



# We can use the pre-trained Mask RCNN model to obtain higher performance
work_dir = 'work_dirs/coco_raw/hoyer_frcnn_lr_test'
load_from = 'work_dirs/coco_raw/hoyer_frcnn_lr_test/epoch_6_12.4.pth'
# load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth' # Ori mAP: 39.9
# load_from = 'work_dirs/coco_raw/frcnn_lr_0.02_without_pretrain/epoch_11.pth'
# bash tools/dist_test.sh configs/bdd100k/coco_raw_frcnn.py work_dirs/coco_raw/frcnn/gpu4/epoch_12.pth 4 --eval bbox

# bash tools/dist_test.sh configs/bdd100k/coco_raw_frcnn.py work_dirs/coco_raw/frcnn/gpu4/epoch_12.pth 4 --eval bbox
