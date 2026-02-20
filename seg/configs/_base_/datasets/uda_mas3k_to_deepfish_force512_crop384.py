# dataset settings for MAS3K -> Deepfish binary semantic segmentation
# variant: force all images to 512x512, then crop to 384x384

dataset_type = 'BinarySegDataset'
source_root = r'B:/3_exp/code_exp/data/MAS3K/fold1/train/'
target_root = r'B:/3_exp/code_exp/data/Deepfish/fold1/train/'
source_valid_root = r'B:/3_exp/code_exp/data/MAS3K/fold1/valid/'
target_valid_root = r'B:/3_exp/code_exp/data/Deepfish/fold1/valid/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (384, 384)

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary_label=True, label_threshold=128, keep_ignore_label=False),
    # NOTE: keep_ratio=False intentionally distorts to fixed 512x512.
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary_label=True, label_threshold=128, keep_ignore_label=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type=dataset_type,
            data_root=source_root,
            img_dir='high',
            ann_dir='Mask',
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            data_root=target_root,
            img_dir='high',
            ann_dir='Mask',
            pipeline=target_train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=target_valid_root,
        img_dir='high',
        ann_dir='Mask',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=target_valid_root,
        img_dir='high',
        ann_dir='Mask',
        pipeline=test_pipeline))
