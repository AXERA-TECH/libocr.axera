import os
import gradio as gr
import numpy as np
from pyocr import AXOCR
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
import cv2
import glob
import argparse
import subprocess
import re
from PIL import Image, ImageDraw, ImageFont

def get_all_local_ips():
    result = subprocess.run(['ip', 'a'], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有IPv4
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)

    # 过滤掉回环地址
    real_ips = [ip for ip in ips if not ip.startswith('127.')]

    return real_ips

def draw_box_and_text(img, box, text, color=(0, 255, 0), thickness=2, font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", font_size=20):
    """
    Draw rotated box and Chinese text using PIL + OpenCV.
    - img: numpy OpenCV image (BGR)
    - box: your Box structure (with center.x/y, size.w/h, angle)
    - text: Unicode string (may contain Chinese)
    - font_path: path to .ttf that supports Chinese (e.g., simhei.ttf)
    """
    # ---- draw box (same as before) ----
    center = (float(box.center_x), float(box.center_y))
    size   = (float(box.w), float(box.h))
    angle  = float(box.angle)
    rect = (center, size, angle)
    pts = cv2.boxPoints(rect)
    pts = np.int0(np.round(pts))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    # ---- now draw Chinese text using PIL ----
    # Convert to RGB for PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Load a Chinese font (you must have the .ttf file)
    font = ImageFont.truetype(font_path, font_size)

    # Draw text near the top-left corner of the box
    x, y = pts[0]
    draw.text((x, y - font_size), text, font=font, fill=(255, 0, 0))  # red text

    # Convert back to OpenCV BGR
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--det', type=str)
    parser.add_argument('--cls', type=str)
    parser.add_argument('--rec', type=str)
    parser.add_argument('--dict', type=str)
    args = parser.parse_args()

    # 初始化
    dev_type = AxDeviceType.unknown_device
    dev_id = -1
    devices_info = enum_devices()
    print("可用设备:", devices_info)
    if devices_info['host']['available']:
        print("host device available")
        sys_init(AxDeviceType.host_device, -1)
        dev_type = AxDeviceType.host_device
        dev_id = -1
    elif devices_info['devices']['count'] > 0:
        print("axcl device available, use device-0")
        sys_init(AxDeviceType.axcl_device, 0)
        dev_type = AxDeviceType.axcl_device
        dev_id = 0
    else:
        raise Exception("No available device")

 
    ocr = AXOCR(
        det_model_path=args.det,
        cls_model_path=args.cls,
        rec_model_path=args.rec,
        rec_charset_path=args.dict,
        dev_type=dev_type,
        devid=dev_id,
    )
    
    def detect_image(img):
        results = ocr.detect(img)
        for obj in results:
            box = obj.rbox
            img = draw_box_and_text(img, box, obj.rec_text, color=(0, 0, 255), thickness=1)
        
        return img


    # Gradio界面
    with gr.Blocks() as demo:
        gr.Markdown("# 🔍 Det Demo")

        with gr.Row():
            input_image = gr.Image(label="输入图像")
            output_image = gr.Image(label="输出图像")
        det_btn = gr.Button("Detect")


        det_btn.click(fn=detect_image, inputs=[input_image], outputs=[output_image])

    # 启动
    ips = get_all_local_ips()
    for ip in ips:
        print(f"* Running on local URL:  http://{ip}:7860")
    ip = "0.0.0.0"
    demo.launch(server_name=ip, server_port=7860)
    
    
    del ocr
    
    import atexit
    if devices_info['host']['available']:
        atexit.register(lambda: sys_deinit(AxDeviceType.host_device, -1))
    elif devices_info['devices']['count'] > 0:
        atexit.register(lambda: sys_deinit(AxDeviceType.axcl_device, 0))
    
    
