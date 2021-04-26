_base_ = './model_r3d18.py'
model = dict(
    backbone=dict(type='R2Plus1D'),
)
