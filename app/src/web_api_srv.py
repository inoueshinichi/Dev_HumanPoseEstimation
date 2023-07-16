"""姿勢推定用のWebAPIサーバー
"""
from email.charset import BASE64
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import re
import io
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
)
import numpy as np
import scipy as sp
# import cv2

from flask import (
    Flask, 
    jsonify, 
    request,
)



import torch
import torchvision

# import sanic
# from sanic import Sanic
# from sanic.response import (
#     json,
#     text,
# )
# from sanic.log import logger
# from sanic.exceptions import ServerError


from type_hint import *
from log_conf import logging

log = logging.getLogger(__name__)

# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

app = Flask(__name__)


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

# REST_API server
app = Flask("rest-api app server")

# 日本語を使えるように
app.config['JSON_AS_ASCII'] = False

API_VER = 'v1'
DB_STORE_COUNT = 1000000 # 100万

# POSTで公開
@app.route(f'/api/{API_VER}/predict/', methods=['POST'])
def api_predict():
    # content_type: str = request.headers.get('Content-Type')
    # if content_type == 'application/json':

    print('[STATUS] arrival request data')

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
    # print('request.json["images"]', images, flush=True)

    for ndx, content in enumerate(images):
        meta = content['meta']
        
        # Image attributes
        filename = meta['filename']
        type = meta['type']
        shape = tuple(meta['shape'])
        print('filename', filename, flush=True)
        print('type', type, flush=True)
        print('shape', shape, flush=True)
        channels = shape[0]
        height = shape[1]
        width = shape[2]

        """ ネットワーク間のバイナリデータの送受信フロー
            Local Host -> Binary -> Base64 -> UTF-8 -> Network -> UTF-8 -> Base64 -> Binary -> Remote Host
            Base64: Base64エンコード方式の文字列 
            UTF-8: UTF-8方式の文字列
            文字列と文字コードの変換 : (Binary - (Encode) -> Base64/UTF-8 -> (Decode) -> Binary)
        """
        
        # utf-8 str 
        img_str = content['data'] # image file string (encoded with base64)

        # utf-8 str -> base64 str
        img_base64 = img_str.encode('utf-8') # base64 encoded str

        # base64 str -> raw binary of image file
        img_bytes = base64.decodebytes(img_base64) 

        # raw binary image file -> pillow
        img_pil = Image.open(io.BytesIO(img_bytes)) # (H,W,C)

        # camera info
        camera_info = {}
        exif = img_pil.getexif()
        for key in exif.keys():
            tag = ExifTags.TAGS.get(key, key)
            camera_info[tag] = exif[key]

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
        
        camera_info = format_bytes_to_str(camera_info)
        print('camera_info', camera_info, flush=True)

        orient = camera_info['Orientation']
        
        # pil -> numpy
        img_np = np.array(img_pil, dtype=np.uint8)
        print('img_np.shape', img_np.shape, flush=True)

        # (H,W,C) -> (C,H,W)
        img_np = img_np.reshape(-1, height, width)
        # img_np = img_np[:3] # until 3 chanells (RGB)

        print('img_np.shape', img_np.shape, flush=True)

        img_np = img_np[:3] # to RGB

        # numpy -> tensor
        in_tensor = torch.from_numpy(img_np)
        # in_tensor = in_tensor.view(*shape)
        in_tensor = in_tensor.to(torch.float32)
        in_tensor = (in_tensor - in_tensor.min()) / (in_tensor.max() - in_tensor.min()) # [0,255] -> [0,1]

        ###########
        # 推論処理 #
        ###########

        # (C,H,W) -> (H,W,C)
        img_np = img_np.reshape(height, width, -1)

        # numpy -> pil
        img_pil = Image.fromarray(img_np)
        
        # pil -> raw binary of image file
        img_bytes = io.BytesIO()
        img_format = type.split('/')[1] # e.g. mime_type: 'image/jpeg' -> 'jpeg'
        img_pil.save(img_bytes, format=img_format)
        img_bytes = img_bytes.getvalue() # bytes

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


if __name__ == "__main__":
    print('[Start] flask app server')
    app.run(debug=True,
            host='0.0.0.0',
            port=APP_PORT,
            )