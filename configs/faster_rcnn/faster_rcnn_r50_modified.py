_base_ = './modified_faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ('electrical-components', 'and', 'antenna', 'capacitor-polarized', 'capacitor-unpolarized', 'crossover', 'diac', 'diode', 'diode-light_emitting', 'fuse', 'gnd', 'inductor', 'integrated_circuit', 'integrated_cricuit-ne555', 'junction', 'lamp', 'microphone', 'motor', 'nand', 'nor', 'not', 'operational_amplifier', 'optocoupler', 'or', 'probe-current', 'relay', 'resistor', 'resistor-adjustable', 'resistor-photo', 'schmitt_trigger', 'socket', 'speaker', 'switch', 'terminal', 'text', 'thyristor', 'transformer', 'transistor', 'transistor-photo', 'triac', 'varistor', 'voltage-dc', 'voltage-dc_ac', 'voltage-dc_regulator', 'vss', 'xor')