_base_ = [
    'pointpillars_hv_secfpn_8xb6-160e_kitti-stereo-pcl-3d-3class.py'
]

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=5)
# custom_hooks = [
#     dict(type='SimpleCheckpoint', interval=16000, artifact_name='model')
# ]