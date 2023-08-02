"""OpenPoseによる人物姿勢推定モジュール
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

########### Standard ###########
import time
import argparse
import datetime
# import hashlib


########### 3rd-parth ###########
import numpy as np
# import cv2

import torch
import torchvision
import torch.nn as nn
# from torch.optim import SGD, Adam
from torch.utils.data import DataLoader


########### Own ###########
from type_hint import *


########### Logging ###########
from log_conf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class OpenPosePredictor:

    def __init__(self,
                 sys_argv=None,
                 ):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for inferencing.',
                            default=1,
                            type=int,
                            )
        
        parser.add_argument('--num-workers',
                            help='Number of worker processes for inferencing.',
                            default=1,
                            type=int,
                            )
        
        parser.add_argument('--model-path',
                            help='What to model file path to use.',
                            action='store',
                            )
        
        parser.add_argument('--weights-path',
                            help='What to weights file path to use.',
                            action='store',
                            )
        
        parser.add_argument('--width',
                            help='What to width of input image.',
                            type=int,
                            )
        
        parser.add_argument('--height',
                            help='What to height of input image.',
                            type=int,
                            )
        
        parser.add_argument('comment',
                            help='Comment for OpenPosePredictor class.',
                            nargs='?',
                            default='Please your comment for OpenPosePredictor',
                            )
        
        # Read arguments
        self.cli_args = parser.parse_args(sys_argv)

        # Check available gpu device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Image properies
        self.width = self.cli_args.width
        self.height = self.cli_args.height
        
        # OpenPose model
        self.model = None
        self.weights = None

        # モデルの初期化
        self.init_model()

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        print("{} setting datetime: {}", __class__.__name__, self.time_str)

        
    def init_model(self):
        """モデルの初期化

        Args:
            filepath (str): モデルファイルへのパス
        """
        # モデルファイルの読み込み
        self.model = torch.load(self.cli_args.model)

        # パラメータファイルの読み込み, CPU形式のパラメータとして読み込む
        self.weights = torch.load(self.cli_args.weights_path, 
                             map_location='cpu',
                             )

        # モデルにパラメータをローディング
        self.model.load_state_dict(self.weights)

        # GPUが使えるならラッピング
        if self.use_cuda:
            log.info('Using CUDA; {} device.'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

    def preprocess(self,
                   x_tensor: torch.Tensor,
                   ) -> torch.Tensor:
        """入力画像の前処理とデータ拡張

        Args:
            x_tensor (torch.Tensor): 入力画像 RGB(3,H,W)

        Returns:
            torch.Tensor: 処理画像 RGB(3,H,W)
        """

        # 正規化
        in_tensor = (x_tensor - x_tensor.min()) / (x_tensor.max() - x_tensor.min()) # [0,255] -> [0,1]

        # 標準化

        # 色々やる...

        return in_tensor

    
    def predict(self,
                src_img: np.ndarray,
                ) -> np.ndarray:
        """OpenPose(Pytorch)による人物姿勢推定

        Args:
            src_img (np.ndarray): 入力画像 RGB(H,W,3)

        Returns:
            np.ndarray: 推定結果画像 RGB(H,W,3)
        """
        # (H,W,C) -> (C,H,W)
        height, width, _ = src_img.shape
        img_np = src_img.reshape(-1, height, width)
        channels = img_np.shape[0]
        log.debug('Convert image shape ({},{},{}) -> ({},{},{})'.format(
            *(height, width, channels), *(channels, height, width)
        ))

        # numpy -> tensor
        in_tensor = torch.from_numpy(src_img)
        in_tensor = in_tensor.to(torch.float32)

        # 前処理
        x_tensor = self.preprocess(in_tensor)

        # 推論 : OpenPoseは10fps程度の処理速度なので, 
        # ここでは100msスリープを入れてレスポンスタイムを擬似的に作る
        time.sleep(0.1) # 100ms

        out_tensor = torch.copy(x_tensor)

        out_img = out_tensor.numpy()

        # (C,H,W) -> (H,W,C)
        out_img = out_img.reshape(height, width, -1)
        log.debug('Return shape (C,H,W)->(H,W,C): ({},{},{})'.format(
            out_img.shape[0],
            out_img.shape[1],
            out_img.shape[2]
            ))

        return out_img
    
