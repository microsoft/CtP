_base_ = '../r3d_18_ucf101/finetune_ucf101.py'

work_dir = './output/ctp/r2plus1d_18_ucf101/finetune_ucf101/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/ctp/r2plus1d_18_ucf101/pretraining/epoch_300.pth',
    ),
)
