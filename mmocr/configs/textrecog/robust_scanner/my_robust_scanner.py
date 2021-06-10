
#自定义robust_scanner配置,只训练所给数据集

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_models/robust_scanner.py'
]

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
total_epochs = 200

img_norm_cfg =  dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=160,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio'
                ]),
        ])
]

dataset_type = 'OCRDataset'

train_prefix = '/workspace/str/recset/'
train_ann_file = train_prefix + 'train_label.txt'
train_base = dict(
    type=dataset_type,
    img_prefix=train_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

train = {key: value for key, value in train_base.items()}
train['img_prefix'] = train_prefix
train['ann_file'] = train_ann_file

test_prefix = '/workspace/str/recset/'
test_ann_file = test_prefix + 'test_label.txt'
test_base = dict(
    type=dataset_type,
    img_prefix=test_prefix,
    ann_file=test_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    test_mode=False)

test = {key: value for key, value in test_base.items()}
test['img_prefix'] = test_prefix
test['ann_file'] = test_ann_file

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[train]
        ),
    val=dict(
        type='ConcatDataset',
        datasets=[test]),
    test=dict(
        type='ConcatDataset',
        datasets=[test]))

evaluation = dict(interval=5, metric='acc')
