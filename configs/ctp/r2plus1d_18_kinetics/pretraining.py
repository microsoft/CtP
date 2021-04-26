_base_ = '../r3d_18_kinetics/pretraining.py'

work_dir = './output/ctp/r2plus1d_18_kinetics/pretraining/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
    )
)


