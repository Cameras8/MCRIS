
import gradio as gr
import cv2
import numpy as np
import os
from ultralytics import YOLO

# 设置上传和结果文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载模型
model = YOLO('best.pt')
def process_image(image):
    # 保存上传的图像
    filename = 'uploaded_image.jpg'
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(file_path, image)

    # 处理图像
    results = model(image)
    detection_results = []
    class_counts = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            class_name = model.names[int(cls)]

            # 计算每种类别的数量
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

            # 绘制边界框和标签
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{class_name}:{conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)
            detection_results.append(f'Class: {class_name}, Confidence: {conf:.2f}, Box: ({x1}, {y1}), ({x2}, {y2})')

    # 在图像上显示检测到的物体数量信息
    y_offset = 30
    for class_name, count in class_counts.items():
        cv2.putText(image, f'{class_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        y_offset += 30

    # 保存处理后的图像
    result_filename = 'result_image.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, image)

    return image, '\n'.join(detection_results)

# 创建Gradio界面
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="上传图像"),
    outputs=[gr.Image(type="numpy", label="处理后的图像"), gr.Textbox(label="检测结果")],
    title="YOLOv11 井盖状态图像检测",
    description="上传图像并使用YOLOv11模型进行检测"
)



# # 启动Gradio应用
iface.launch(
    server_name="1.94.232.138",  # 服务器IP地址
    server_port=9999,             # 使用的端口
    debug=True,                   # 可选：启用调试模式
    auth=("username", "password"),# 可选：设置基本认证
    auth_message="Please enter your credentials to access the app.", # 可选：自定义认证消息
    inbrowser=False               # 可选：是否在服务器启动后自动打开浏览器
)