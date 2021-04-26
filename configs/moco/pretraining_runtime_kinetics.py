_base_ = './pretraining_runtime_ucf.py'

data = dict(
    train=dict(
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='kinetics400/train.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='kinetics400/{}/RGB_frames.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
    )
)

# optimizer
total_epochs = 90
