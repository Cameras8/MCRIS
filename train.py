from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLOv11 model
    model = YOLO(r'yolov11/model/yolov8n.pt')

    # Train the model on a custom dataset
    model.train(
        data='/root/yolov11/datasets/data/data.yaml',  # 数据集配置文件路径
        epochs=150,  # 训练轮数
        imgsz=640,  # 输入图像大小
        batch=16,  # 批量大小
        workers=8,  # 数据加载线程数
        project='/root/yolov11/runs',  # 指定输出目录为 yolov11 下的 runs 文件夹
        name='train',
        pretrained=False
    )
