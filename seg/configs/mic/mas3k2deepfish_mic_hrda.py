# MIC(HRDA) config for MAS3K -> Deepfish binary semantic segmentation

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_mas3k_to_deepfish_512x512.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 2

model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(
        type='HRDAHead',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1,
        num_classes=2),
    scales=[1, 0.5],
    hr_crop_size=(256, 256),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[256, 256],
        crop_size=[512, 512]))

uda = dict(
    alpha=0.999,
    pseudo_threshold=0.968,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True))

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

name = 'mas3k2deepfish_mic_hrda_s2'
exp = 'custom'
name_dataset = 'mas3k2deepfish_512x512'
name_architecture = 'hrda1-256-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-256-0.1_daformer_sepaspp_sl'
name_uda = 'dacs_a999_mic_m64-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
