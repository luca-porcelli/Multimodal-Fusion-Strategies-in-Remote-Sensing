import os
from mmseg.models import build_segmentor
from mmcv.utils import Config
from mmcv.utils import get_logger
from mmcv.cnn.utils import revert_sync_batchnorm
from torch import nn
from os.path import dirname as up
from new import BilinearConvTranspose2d

logger = get_logger('mmdet')
logger.setLevel('WARNING')

#configs_path = os.path.join(up(__file__), 'configs')

class MariNext(nn.Module):

    def __init__(self, in_chans, num_classes):
        super(MariNext, self).__init__()
        conf_file = os.path.join("/home/ubuntu/Tesi/Early-Fusion/Model/marinext.tiny.240x240.mados.py")
        cfg = Config.fromfile(conf_file)
        cfg.model.backbone.in_chans = in_chans
        cfg.model.decode_head.num_classes = num_classes
        model = build_segmentor(cfg.model)
        model.init_weights()
        model = revert_sync_batchnorm(model)
        
        self.backbone = model.backbone
        self.decode_head = model.decode_head 
        #self.layer_upsample = BilinearConvTranspose2d(15,4,15)
        #for param in self.layer_upsample.parameters():
        #    param.requires_grad = False
        #self.layer_upsample = nn.ConvTranspose2d(in_channels=15, out_channels=15, kernel_size=4, stride=4)

    def forward(self, x):
        #return self.layer_upsample(self.decode_head(self.backbone(x)))
        return self.decode_head(self.backbone(x))