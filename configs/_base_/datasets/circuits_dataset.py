# dataset settings
dataset_type = 'CocoDataset'
data_root = '/content/CGHD-coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
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
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ('electrical-components', 'and', 'antenna', 'capacitor-polarized', 'capacitor-unpolarized', 'crossover', 'diac', 'diode', 'diode-light_emitting', 'fuse', 'gnd', 'inductor', 'integrated_circuit', 'integrated_cricuit-ne555', 'junction', 'lamp', 'microphone', 'motor', 'nand', 'nor', 'not', 'operational_amplifier', 'optocoupler', 'or', 'probe-current', 'relay', 'resistor', 'resistor-adjustable', 'resistor-photo', 'schmitt_trigger', 'socket', 'speaker', 'switch', 'terminal', 'text', 'thyristor', 'transformer', 'transistor', 'transistor-photo', 'triac', 'varistor', 'voltage-dc', 'voltage-dc_ac', 'voltage-dc_regulator', 'vss', 'xor')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        classes=classes,
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/valid.json',
        img_prefix=data_root + 'valid/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
