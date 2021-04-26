_base_ = './runtime_ucf101.py'

data = dict(
    train=dict(
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='something-something-v1/train.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='something-something-v1/{}/RGB_frames.zip',
            frame_fmt='{:05d}.jpg',
        ),
        frame_sampler=dict(
            type='RandomFrameSampler',
            num_clips=1,
            clip_len=32,
            strides=1,
            temporal_jitter=False
        ),
        test_mode=False,
        transform_cfg=[
                dict(type='GroupScale', scales=[112, 128, 144]),
                dict(type='RandomBrightness', prob=0.20, delta=32),
                dict(type='RandomContrast', prob=0.20, delta=0.20),
                dict(type='RandomHueSaturation', prob=0.20, hue_delta=12, saturation_delta=0.1),
                dict(type='GroupRandomCrop', out_size=112),
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=True,
                    div255=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
    ),
    val=dict(
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='something-something-v1/val.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='something-something-v1/{}/RGB_frames.zip',
            frame_fmt='{:05d}.jpg',
        ),
        frame_sampler=dict(
            type='UniformFrameSampler',
            num_clips=1,
            clip_len=32,
            strides=1,
            temporal_jitter=False
        ),
        test_mode=True,
        transform_cfg=[
                dict(type='GroupScale', scales=[128]),
                dict(type='GroupCenterCrop', out_size=112),
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=True,
                    div255=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
    ),
    test=dict(
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='something-something-v1/val.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='something-something-v1/{}/RGB_frames.zip',
            frame_fmt='{:05d}.jpg',
        ),
        frame_sampler=dict(
            type='UniformFrameSampler',
            num_clips=1,
            clip_len=32,
            strides=1,
            temporal_jitter=False
        ),
        test_mode=True,
        transform_cfg=[
                dict(type='GroupScale', scales=[128]),
                dict(type='GroupCenterCrop', out_size=112),
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=True,
                    div255=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
    )
)

total_epochs = 90
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4)
lr_config = dict(
    policy='step',
    step=[40, 60, 80],
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
)
