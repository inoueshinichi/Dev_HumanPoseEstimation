"""PoseNetによる人物姿勢推定
"""
from configparser import Interpolation
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
from collections import namedtuple

########### 3rd-parth ###########
import numpy as np
# import cv2

# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter
# pylint: enable=g-import-not-at-top


from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

########### Own ###########
from type_hint import *

########### Logging ###########
from log_conf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


PoseNetHumanPoseKeypoints = namedtuple(
    'HumanPoseKeypoints',
    [
        'nose',           # 0
        'left_eye',       # 1
        'right_eye',      # 2
        'left_ear',       # 3
        'right_ear',      # 4
        'left_shoulder',  # 5
        'right_shoulder', # 6
        'left_elbow',     # 7
        'right_elbow',    # 8
        'left_wrist',     # 9
        'right_wrist',    # 10
        'left_hip',       # 11
        'right_hip',      # 12
        'left_knee',      # 13
        'right_knee',     # 14
        'left_ankle',     # 15
        'right_ankle'     # 16
    ]
)

class PoseNetPredictor:

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
                            default=2,
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

        # Model file
        _, ext = os.path.splitext(self.cli_args.model_path)
        if ext is not 'tflite':
            raise ValueError(f'Extension of filename must be tflite. Given is {ext}.')

        # Image properies
        self.width = self.cli_args.width
        self.height = self.cli_args.height

        # モデルの初期化
        self.init_model()

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        print("{} setting datetime: {}", __class__.__name__, self.time_str)

    def init_model(self):
        """モデルの初期化

        Args:
            filepath (str): モデルファイルへのパス
        """

        # Initialize model
        interpreter = Interpreter(model_path=self.cli_args.model_path, 
                                  num_threads=self.cli_args.num_worker,
                                  )
        interpreter.allocate_tensors()

        # 入力形状
        self._input_index = interpreter.get_input_details()[0]['index']
        # self._input_channels = interpreter.get_input_details()[0]['shape'][0]
        self._input_height = interpreter.get_input_details()[0]['shape'][1]
        self._input_width = interpreter.get_input_details()[0]['shape'][2]

        # 出力形状
        self._output_heatmap_index = interpreter.get_output_details()[0]['index']
        self._output_offset_index = interpreter.get_output_details()[1]['index']

        self._interpreter = interpreter


    def _preprocess(self,
                   in_tensor: np.ndarray,
                   ) -> np.ndarray:
        """前処理
        1. 画像のりサイズ
        2. 標準化

        Args:
            x_tensor (np.ndarray): _description_
        """
        
        x_tensor = cv2.resize(in_tensor,
                             (self._input_width, 
                              self._input_height))
        x_tensor = np.expand_dims(x_tensor, axis=0) # (N,H,W,C)

        # check the type of the input tensor
        is_float_model = self._interpreter.get_input_details()[0]['dtype'] == np.float32
        
        if is_float_model:
            x_tensor = (np.float32(x_tensor) - 127.5) / 127.5

        return x_tensor

    def predict(self,
                src_img: np.ndarray,
                ) -> Tuple[np.ndarray, PoseNetHumanPoseKeypoints]:
        """推論

        Args:
            src_img (np.ndarray): RGB画像(H,W,3)

        Returns:
            np.ndarray: _description_
        """

        img_height, img_width, _ = src_img.shape

        in_tensor = self._preprocess(src_img)
        
        # # Create a TensorImage object from the RGB image.
        # src_img = src_img.transpose(2,0,1) # (C,H,W) -> (H,W,C)
        # in_tensor = vision.TensorImage.create_from_array(src_img)

        
        # 前処理済みの画像を入力
        self._interpreter.set_tensor(self._input_index, in_tensor)

        # 推論
        self._interpreter.invoke()

        # インタプリタ(モデル)から出力データを抽出
        raw_heatmap = self._interpreter.get_tensor(self._output_heatmap_index)
        raw_offset = self._interpreter.get_tensor(self._output_offset_index)

        # Getting rid of the extra dimension
        raw_heatmap = np.squeeze(raw_heatmap) # (N,C,H,W) -> (C,H,W)
        raw_offset = np.squeeze(raw_offset) # (N,C,H,W) -> (C,H,W)

        # 関節点の取得
        keypoints_with_scores = self._build_keypoints(raw_heatmap,
                                                      raw_offset,
                                                     )
        keypoints = self._person_from_keypoints_with_scores(keypoints_with_scores,
                                                            img_height,
                                                            img_width,
                                                            )

        
        
        return dst_img, keypoints
    
    def _build_keypoints(self,
                         heatmap: np.ndarray,
                         offset: np.ndarray,
                         ) -> np.ndarray:
        """ヒートマップとオフセットのスコアマップを作成

        Args:
            heatmap (np.ndarray): 各関節(17種類)毎のヒートマップ (H,W,17)
            offset (np.ndarray): 各四肢に対応する2Dベクトル場の特徴マップ (H,W,34)

        Returns:
            np.ndarray: 各関節点のスコアマップ (17,3) index: [(xs,ys), (xe,ye), score]
        """

        # 関節点数
        num_joints = heatmap.shape[-1]

        # スコア表
        keypoints_with_scores = np.zeros((num_joints, 3), np.float32)
        scores = self._sigmoid(heatmap) # (H,W,17) [0,1]

        for idx in range(num_joints):
            # スコアマップの取得
            joint_heatmap = heatmap[..., idx]

            # スコアマップ中の最大値を示す2次元インデックス(idx_x,idx_y)を取得
            x, y = np.unravel_index(np.argmax(scores[:, :, idx]), 
                                    scores[:, :, idx].shape)
            
            # (x,y)を取得
            max_val_pos = np.squeeze(np.argwhere(
                joint_heatmap == np.max(joint_heatmap)))
            
            # リマップ　(まだ理解できてない)
            remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)

            # (xs, ys)
            keypoints_with_scores[idx, 0] = (
                remap_pos[0] + offset[max_val_pos[0], max_val_pos[1], idx]) / 257

            # (xe, ye)
            keypoints_with_scores[idx, 1] = (
                remap_pos[1] + offset[max_val_pos[0], max_val_pos[1], idx + num_joints]) / 257
            
            # スコア
            keypoints_with_scores[idx, 2] = scores[x, y, idx]

        return keypoints_with_scores


    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def _person_from_keypoints_with_scores(self, 
                         keypoints_with_scores: np.ndarray,
                         height: int,
                         width: int,
                         ) -> PoseNetHumanPoseKeypoints:
        """PoseNetで定義されているキーポイント(各関節)を取得する

        Args:
            keypoints_with_scores (np.ndarray): 各関節点のスコアマップ (H,W,17)
        
        Returns:
            PoseNetHumanPoseKeypoints: [x, y]

            PoseNetHumanPoseKeypoints = namedtuple(
            'HumanPoseKeypoints',
                [
                    'nose',           # 0 [x, y]
                    'left_eye',       # 1
                    'right_eye',      # 2
                    'left_ear',       # 3
                    'right_ear',      # 4
                    'left_shoulder',  # 5
                    'right_shoulder', # 6
                    'left_elbow',     # 7
                    'right_elbow',    # 8
                    'left_wrist',     # 9
                    'right_wrist',    # 10
                    'left_hip',       # 11
                    'right_hip',      # 12
                    'left_knee',      # 13
                    'right_knee',     # 14
                    'left_ankle',     # 15
                    'right_ankle'     # 16
                ]
            )
        """
        pass

