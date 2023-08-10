"""姿勢推定用のWebAPIサーバー
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import re
import time
import json
import asyncio
import itertools
import functools
import threading
import base64

import redis
from PIL import (
    Image,
    ExifTags,
    ImageDraw,
    ImageFont,
)
import numpy as np
import scipy as sp
# import cv2

from type_hint import *

pose_estimation_model = "PoseNet"

if pose_estimation_model == "PoseNet":
    # PoseNet
    from posenet.runtime_predict import (
        PoseNetPredictor,
        PoseNetHumanPoseKeypoints,
    )
else:
    # OpenPose
    # from openpose.runtime_predict import (
    #     OpenPosePredictor,
    #     OpenPoseHumanPoseKeypoints,
    # )
    pass


########### Logging ###########
from log_conf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


########### Pose Estimation Model ############
# pose_estimation = PoseNetPredictor()
# pose_estimation.init_model()

########### Flask ###########
from flask import (
    Flask, 
    jsonify, 
    request,
)

# ### sanic ###
# import sanic
# from sanic import Sanic
# from sanic.response import (
#     json,
#     text,
# )
# from sanic.log import logger
# from sanic.exceptions import ServerError

app = Flask(__name__)

# REST_API server
app = Flask("rest-api app server")

# 日本語を使えるように
app.config['JSON_AS_ASCII'] = False

# apiサーバーのバージョン
API_VER = 'v1'

########### Redis kvs ###########
# Redis db conf
REDIS_HOST: str = os.environ['REDIS_HOST']
REDIS_PORT: int = int(os.environ['REDIS_PORT'])
REDIS_DB: int = int(os.environ['REDIS_DB'])

print('---- Check db access setting ----')
print('REDIS_HOST', REDIS_HOST)
print('REDIS_PORT', REDIS_PORT)
print('REDIS_DB', REDIS_DB)

print('Connect redis db server')
REDIS = redis.Redis(host=REDIS_HOST, 
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    )
print('Redis db server: ', REDIS)

# App port (for this REST_API server)
APP_PORT = int(os.environ['APP_PORT'])

DB_STORE_COUNT = 1000000 # 100万


# POSTで公開
@app.route(f'/api/{API_VER}/predict', methods=['POST'])
def api_predict():
    # content_type: str = request.headers.get('Content-Type')
    # if content_type == 'application/json':

    log.info('[START] api_predict, URI: {}'.format(f'/api/{API_VER}/predict'))

    if not request.is_json:
        return error(400.4)

    id: int = request.json['id']
    hostname: str = request.json['host']
    portname: str = request.json['port']
    first_name: str = request.json['first_name']
    last_name: str = request.json['last_name']
    age: int = request.json['age']
    gender: str = request.json['gender'] # '男':0, '女':1, 'その他':2
    gender_category = {
        0: '男',
        1: '女', 
        2: 'その他'
    }


    response_list = []
    images: Any = request.json['images']
    
    # log.debug('requext.json["images"]', images, flush=True)

    for ndx, content in enumerate(images):
        meta = content['meta']
        
        # Image attributes
        filename = meta['filename']
        type = meta['type']
        shape = tuple(meta['shape'])

        # print('filename', filename, flush=True)
        # print('type', type, flush=True)
        # print('shape', shape, flush=True)
        log.debug('filename: {}'.format(filename))
        log.debug('type: {}'.format(type))
        log.debug('shape: {}'.format(shape))

        # channels = shape[0]
        height = shape[0]
        width = shape[1]

        """ ネットワーク間のバイナリデータの送受信フロー
            Local Host -> Binary -> Base64 -> (UTF-8 ->) Network -> (UTF-8 ->) Base64 -> Binary -> Remote Host
            Base64: Base64エンコード方式の文字列 
            UTF-8: UTF-8方式の文字列
            文字列と文字コードの変換 : (Binary - (Encode) -> Base64/UTF-8 -> (Decode) -> Binary)
            Base64とUTF-8の文字コード変換は必要ないのでは???
        """
        
        # utf-8 str 
        img_str = content['data'] # image file string (encoded with base64)

        # utf-8 str -> base64 str
        img_base64 = img_str.encode('utf-8') # base64 encoded str

        # base64 str -> raw binary of image file
        img_bytes = base64.decodebytes(img_base64) 

        # raw binary image file -> pillow
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB") # (H,W,C) alpha値除外
        info_pil = img_pil.info
        log.debug('info_pil: {}'.format(info_pil))
        format_pil = img_pil.format
        log.debug('format_pil: {}'.format(format_pil))


        ############################################
        # iOSで撮影した画像(Jpeg)は, 画像の縦横サイズ情報
        # に対して, 実際の画像が90°回転してしまっている問題
        # があるので対処する.                              
        ############################################
        # 画像情報の取得
        camera_info = {}
        exif = img_pil.getexif() # EXIFが画像情報を持つ対象属性
        if exif is not None:
            for key in exif.keys():
                tag = ExifTags.TAGS.get(key, key)
                camera_info[tag] = exif[key]

            # 整数から文字に変換
            def format_bytes_to_str(camera_info):
                res = {}
                for key, value in camera_info.items():
                    if isinstance(value, bytes):
                        res[key] = "{}".format(value)
                    elif isinstance(value, dict):
                        res[key] = format_bytes_to_str(value)
                    else:
                        res[key] = value
                return res
    
            # カメラに関するInformationgがあれば実行
            camera_info = format_bytes_to_str(camera_info)
            # print('camera_info', camera_info, flush=True)
            log.debug('camera_info: {}'.format(camera_info))
            if camera_info is not None and len(camera_info) != 0:

                """回転情報 : Orientation
                https://kapibara-sos.net/archives/658
                状態
                1: 無補正(No Affine)
                2: 水平フリップ
                3: 水平フリップ&垂直フリップ
                4: 垂直フリップ
                5: 水平フリップ&反時計回りに90°回転
                6: 反時計回りに180°回転
                7: 水平フリップ&時計回りに90°回転
                8: 時計回りに90°回転

                補正方法
                1: なし
                2: 0: Transpose.FLIP_LEFT_RIGHT
                3: 3: Transpose.ROTATE_180
                4: 1: Transpose.FLIP_TOP_BOTTOM
                5: 5: Transpose.TRANSPOSE
                6: 4: Transpose.ROTATE_270
                7: 6: Transpose.TRANSVERSE
                8: 2: Transpose.ROTATE_90

                @warning Pillowには特定のExifタグのみを編集する機能は存在しない.
                """
                # 'Orientation'属性がない場合があるのでチェック
                if 'Orientation' in camera_info.keys():
                    orientation = camera_info['Orientation']
                    if orientation > 1:
                        transpose_to_valid = [0,3,1,5,4,6,2][orientation - 2]
                        img_pil = img_pil.transpose(transpose_to_valid)

        #########
        # 仮の処理
        #########
        # ロゴを入れる
        draw = ImageDraw.Draw(img_pil)
        top = 0
        left = 0
        logo_width = width
        right = left + logo_width if (left + logo_width) < width else width - 1
        logo_height = int(height * 0.2)
        bottom = top + logo_height if (top + logo_height) < height else height - 1
        left_top = (left, top)
        right_bottom = (right, bottom)

        # 背景(黒ベタ)
        draw.rectangle((left_top, right_bottom), fill=(0, 0, 0))

        # 文字列
        # font = ImageFont.truetype(font=None, size=24)
        draw.text((0, 0), 'Tashiro Club', 'white', align='center')#, font=font)
        
        # pil -> numpy
        img_np = np.array(img_pil, dtype=np.uint8)
        # print('img_np.shape', img_np.shape, flush=True)
        log.debug('img_np.shape: {}'.format(img_np.shape))

        # チャネルを取得
        channels = img_np.shape[-1] # (H,W,C)

        # Alpha値を除外する
        img_np = img_np[:3] # until 3 chanells (RGB)

        ###########
        # 推論処理 #
        ###########
        # if pose_estimation_model == "PoseNet":
        #     dst_img = pose_estimation.predict(img_np)


           
        if channels == 4: # RGBA
            # Alpha値を追加
            alpha = np.ones((height, width, 1), dtype=np.uint8)
            log.debug('Alpha: {}'.format(alpha.shape))
            img_np = np.concatenate([alpha, img_np], axis=-1)
            log.debug('Add alpha channels: {}'.format(img_np.shape))

        # numpy -> pil
        # img_pil = Image.fromarray(img_np)
        img_pil.frombytes(img_np.tobytes()) # 入力meta情報を持ったまま画像データのみ入れ替え

        
        
        # pil -> raw binary of image file
        img_format = type.split('/')[1] # e.g. mime_type: 'image/jpeg' -> 'jpeg'
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format=img_format)
        img_bytes = img_bytes.getvalue() # bytes of image file

        log.debug('img_pil info: {}'.format(img_pil.info))
        log.debug('img_pil format: {}'.format(img_pil.format))


        # raw binary of image file -> base64 str
        img_base64 = base64.encodebytes(img_bytes) # base64 encoded image str

        # base64 str -> utf-8
        img_str = img_base64.decode('utf-8') # base64 encoded image string

        json_element = {
            'index': ndx,
            'filename': filename,
            'type': type,
            'data': img_str
        }

        response_list.append(json_element)

    res_data = {
        'images': response_list
    }
    

    log.info('[END] api_predict, URI: {}'.format(f'/api/{API_VER}/predict'))

    return success(res_data)


@app.route(f'/api/{API_VER}/keys/', methods=['GET'])
def api_keys():
    data = {}
    cursor = '0'
    while cursor != 0:
        cursor, keys = REDIS.scan(cursor=cursor,
                                  count=DB_STORE_COUNT,
                                  )
        if len(keys) == 0:
            break
        keys = [key.decode() for key in keys]
        values = [value.decode() for value in REDIS.mget(*keys)]
        data.update(dict(zip(keys, values)))
    
    return success(data)


@app.route(f'/api/{API_VER}/keys/<key>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_key(key):
    if not isalnum(key):
        print('key is not isalnum')
        return error(400.1)
    
    body = request.get_data().decode().strip()
    if request.method in ['POST', 'PUT']:
        if body == '':
            print('body empty')
            return error(400.2)
        if not isalnum(body):
            print('body is not alnum')
            return error(400.3)
    
    def get():
        value = REDIS.get(key)
        if value is not None: # valueが空
            return success({key:value.decode()})
        return error(404)
    
    def post():
        if REDIS.get(key) is not None: # 既にkeyが存在
            return error(409)
        REDIS.set(key, body)
        return success({key:body})
    
    def put():
        REDIS.set(key, body)
        return success({key:body})
    
    def delete():
        if REDIS.delete(key) == 0: # keyが空
            return error(404)
        return success({})
    
    func_dict = {
        'GET': get,
        'POST': post,
        'PUT': put,
        'DELETE': delete,
    }
    return func_dict[request.method]() # HTTPのAPIに応じてRedisとの仲介を行う


# 404 レスポンスコード(アクセスデータが存在しない) ハンドラ
@app.errorhandler(404)
def api_not_found_error(error):
    return jsonify({'error': "api not found", 'code': 404}), 404

# 405 レスポンスコード(アクセス権限) ハンドラ
@app.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({'error': "method not allowed", 'code': 405}), 405

# 500 レスポンスコード(サーバー内部エラー) ハンドラ
@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'server internal error', 'code': 500}), 500


def isalnum(text):
    # 半角英数字で構成された文字列にマッチした場合, Trueを返す
    return re.match(r'^[a-zA-Z0-9]+$', text) is not None

def success(kv):
    # ブラウザにJson形式で値を返す
    return jsonify(kv), 200 # HTTPレスポンスコード 200 : 成功

def error(code):
    message = {
        400.1: "bad request. key must be alnum",
        400.2: "bad request. post/put needs value on body",
        400.3: "bad request. value must be alnum",
        400.4: "bad request. content-type must be application/json",
        404: "request not found",
        409: "resource conflict. resource already exist",
    }
    return jsonify({'error': message[code], 'code': int(code)}), int(code)








def predict_by_posenet(src_img: np.ndarray) -> np.ndarray:
    """PoseNet(Tensorflow Lite)による人物姿勢推定

    Args:
        src_img (np.ndarray): 入力画像 (3,H,W)

    Returns:
        np.ndarray: 出力画像 (3,H,W)
    """
    # numpy -> tensor

    # 推論



if __name__ == "__main__":
    print('[Start] flask app server')
    app.run(debug=True,
            host='0.0.0.0',
            port=APP_PORT,
            )