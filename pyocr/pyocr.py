import ctypes
import os
from typing import List, Tuple
import numpy as np
import platform
from pyaxdev import _lib, AxDeviceType, AxDevices, check_error


class OCRInit(ctypes.Structure):
    _fields_ = [
        ('dev_type', AxDeviceType),
        ('devid', ctypes.c_char),
        ('det_model_path', ctypes.c_char * 256),
        ('cls_model_path', ctypes.c_char * 256),
        ('rec_model_path', ctypes.c_char * 256),
        ('rec_charset_path', ctypes.c_char * 256),
    ]

class OCRImage(ctypes.Structure):
    _fields_ = [
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('channels', ctypes.c_int),
        ('stride', ctypes.c_int),
        ('data', ctypes.POINTER(ctypes.c_ubyte)),
    ]


class Center(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
    ]

class Size(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
    ]

class Box(ctypes.Structure):
    _fields_ = [
        ("center", Center),
        ("size", Size),
        ("angle", ctypes.c_float),
    ]

class Token(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_int),
        ("score", ctypes.c_float),
    ]

class ObjectItem(ctypes.Structure):
    _fields_ = [
        ("box", Box),
        ("score", ctypes.c_float),
        ("orientation", ctypes.c_int),
        ("tokens", Token * 256),
        ("num_tokens", ctypes.c_int),
        ("text", ctypes.c_char * 256),
    ]
    
class ObjectResult(ctypes.Structure):
    _fields_ = [
        ('objects', ObjectItem * 64),
        ('num_objs', ctypes.c_int),
    ]

_lib.ax_ocr_init.argtypes = [ctypes.POINTER(OCRInit), ctypes.POINTER(ctypes.c_void_p)]
_lib.ax_ocr_init.restype = ctypes.c_int

_lib.ax_ocr_deinit.argtypes = [ctypes.c_void_p]
_lib.ax_ocr_deinit.restype = ctypes.c_int

_lib.ax_ocr.argtypes = [ctypes.c_void_p, ctypes.POINTER(OCRImage), ctypes.POINTER(ObjectResult)]
_lib.ax_ocr.restype = ctypes.c_int

class RotatedBox:
    def __init__(self, center_x: int, center_y: int, w: int, h: int, angle: float):
        self.center_x = center_x
        self.center_y = center_y
        self.w = w
        self.h = h
        self.angle = angle

class Object:
    def __init__(self, rbox: RotatedBox, score: float, orientation: int, rec_text: str, tokens: List[Tuple[int, float]] = []):
        self.rbox = rbox
        self.score = score
        self.orientation = orientation
        self.rec_text = rec_text
        self.tokens = tokens

    def __repr__(self):
        return f"Object(rbox={self.rbox}, score={self.score:.2f}, orientation={self.orientation}, rec_text={self.rec_text}, tokens={self.tokens})"
        
class AXOCR:
    def __init__(self, det_model_path: str, 
                 cls_model_path: str, 
                 rec_model_path: str, 
                 rec_charset_path: str, 
                 dev_type: AxDeviceType = AxDeviceType.axcl_device,
                 devid: int = 0):
        self.handle = None
        self.init_info = OCRInit()
        
        # 设置初始化参数
        self.init_info.dev_type = dev_type
        self.init_info.devid = devid
        
        # 设置路径
        self.init_info.det_model_path = det_model_path.encode('utf-8')
        self.init_info.cls_model_path = cls_model_path.encode('utf-8')
        self.init_info.rec_model_path = rec_model_path.encode('utf-8')
        self.init_info.rec_charset_path = rec_charset_path.encode('utf-8')
                
        # 创建OCR实例
        handle = ctypes.c_void_p()
        check_error(_lib.ax_ocr_init(ctypes.byref(self.init_info), ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        if self.handle:
            _lib.ax_ocr_deinit(self.handle)

    def detect(self, image_data: np.ndarray) -> List[Object]:
          
        image = OCRImage()
        image.data = ctypes.cast(image_data.ctypes.data, ctypes.POINTER(ctypes.c_ubyte))
        image.width = image_data.shape[1]
        image.height = image_data.shape[0]
        image.channels = image_data.shape[2]
        image.stride = image_data.shape[1] * image_data.shape[2]
        result = ObjectResult()
        check_error(_lib.ax_ocr(self.handle, ctypes.byref(image), ctypes.byref(result)))
        objects = []
        for i in range(result.num_objs):
            _obj = result.objects[i]
            
            obj = Object(
                rbox=RotatedBox(
                    center_x=_obj.box.center.x,
                    center_y=_obj.box.center.y,
                    w=_obj.box.size.w,
                    h=_obj.box.size.h,
                    angle=_obj.box.angle,
                ),
                score=_obj.score,
                orientation=_obj.orientation,
                rec_text=_obj.text.decode('utf-8'),
                tokens=[(_obj.tokens[j].token, _obj.tokens[j].score) for j in range(_obj.num_tokens)],
            )
            objects.append(obj)
        return objects