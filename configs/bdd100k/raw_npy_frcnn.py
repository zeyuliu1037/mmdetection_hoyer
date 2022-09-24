# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=True),
        norm_eval=False,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromNpy'),
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
    train=dict(
        pipeline=train_pipeline,
        img_prefix='/nas/home/gdatta/raw_image_contents/demosaiced_image/',
        ann_file='/nas/home/zeyul/mmdetection/data/ann/instances_maxitrainX2.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='/nas/home/gdatta/raw_image_contents/demosaiced_image/',
        ann_file='/nas/home/zeyul/mmdetection/data/ann/instances_minivalX2.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='/nas/home/zeyul/mmdetection/data/raw_dataset/RAW_npy',
        ann_file='/nas/home/zeyul/mmdetection/data/raw_dataset/RAW_npy/ann/test_all_gt_annX2.json'))

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

optimizer_config = dict(
     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# Modify dataset related settings



# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'nas/home/shunlin/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
work_dir = 'work_dirs/coco_raw/frcnn/gpu4'
load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth' # Ori: mAP: 39.9

# bash tools/dist_test.sh configs/bdd100k/coco_raw_frcnn.py work_dirs/coco_raw/frcnn/gpu4/epoch_12.pth 4 --eval bbox

# bash tools/dist_test.sh configs/bdd100k/raw_npy_frcnn.py work_dirs/coco_raw/frcnn/gpu4/epoch_12.pth 4 --eval bbox

# bash tools/dist_test.sh configs/bdd100k/raw_npy_frcnn.py checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth 4 --eval bbox
