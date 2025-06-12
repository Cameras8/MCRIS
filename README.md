# 🔍 MCRIS - 智能井盖风险识别系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Latest-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-91.6%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*基于深度学习的城市基础设施安全监测解决方案*

</div>

## 📖 项目简介

MCRIS（Manhole Cover Risk Identification System）是一个基于YOLOv11深度学习框架开发的智能井盖风险识别系统。该系统致力于通过计算机视觉技术，自动识别和分类城市道路中井盖的安全状态，为城市基础设施安全管理提供智能化解决方案。

### 🎯 核心特性

- **高精度识别**：基于YOLOv11架构，模型精度达到91.6%
- **多状态分类**：支持5种井盖状态识别（完好、破损、缺失、未覆盖、井圈问题）
- **实时检测**：支持图像实时上传和处理
- **可视化界面**：基于Gradio构建的友好Web界面
- **轻量化部署**：支持多种模型格式导出，便于部署

## 🏗️ 系统架构

```
MCRIS/
├── 📁 datasets/          # 数据集目录
│   └── data/            # 训练数据
│       ├── train/       # 训练集
│       ├── valid/       # 验证集
│       ├── test/        # 测试集
│       └── data.yaml    # 数据配置文件
├── 📁 model/            # 预训练模型
├── 📁 runs/             # 训练结果
│   └── 91.6%版本/       # 最佳模型版本
├── 📁 results/          # 检测结果
├── 📁 uploads/          # 上传文件
├── 🐍 main.py           # 主程序（Web界面）
├── 🐍 train.py          # 模型训练
├── 🐍 test.py           # 模型测试
├── 🐍 export.py         # 模型导出
└── 📄 best.pt           # 最佳训练权重
```

## 🔬 数据集详情

### 数据来源
本项目数据集来源于团队实地调研，涵盖：
- **北京林业大学校园**：覆盖主要道路和人行道
- **半导体所及家属区**：包含不同类型的井盖设施
- **总计样本**：约1700余张高质量井盖图像

### 标注类别
系统支持识别以下5种井盖状态：

| 类别 | 英文标识 | 描述 | 风险等级 |
|------|----------|------|----------|
| 完好 | `good` | 井盖完整无损，位置正确 | 🟢 安全 |
| 破损 | `broke` | 井盖有裂纹、缺角等损坏 | 🟡 中等风险 |
| 缺失 | `lose` | 井盖完全丢失 | 🔴 高风险 |
| 未覆盖 | `uncovered` | 井口未被井盖覆盖 | 🔴 高风险 |
| 井圈问题 | `circle` | 圆形井盖（特殊标识） | 🟢 正常 |

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
CUDA >= 11.0 (可选，用于GPU加速)
```

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-username/MCRIS.git
cd MCRIS

# 安装依赖包
pip install ultralytics
pip install gradio
pip install opencv-python
pip install numpy
```

### 运行系统

#### 1. Web界面模式
```bash
python main.py
```
访问 `http://localhost:9999` 使用Web界面进行图像检测
![image](https://github.com/user-attachments/assets/c208c282-09e3-416a-bd91-30d5f8000e7c)

#### 2. 命令行测试
```bash
python test.py --cfg yolov8n.pt
```

#### 3. 模型训练
```bash
python train.py
```

## 📊 性能指标

### 模型性能
- **整体精度**：91.6%
- **训练轮数**：150 epochs
- **批处理大小**：16
- **输入图像尺寸**：640×640

### 各类别性能
详细的性能指标可在 `runs/91.6%版本/results.csv` 中查看，包括：
- Precision（精确率）
- Recall（召回率）
- F1-Score
- mAP@0.5
- mAP@0.5:0.95

## 🛠️ 使用指南

### Web界面使用
1. 启动系统：`python main.py`
2. 打开浏览器访问指定地址
3. 上传待检测的井盖图像
4. 查看检测结果和风险评估

### API调用示例
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('best.pt')

# 进行预测
results = model('path/to/your/image.jpg')

# 处理结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_name = model.names[int(box.cls[0])]
        confidence = box.conf[0]
        print(f"检测到: {class_name}, 置信度: {confidence:.2f}")
```

## 🔧 模型导出

支持多种格式导出以适应不同部署需求：

```bash
# 导出为ONNX格式
python export.py

# 支持的格式
# - ONNX
# - TorchScript
# - TensorRT
# - CoreML
```

## 📈 训练自定义模型

### 数据准备
1. 按照YOLO格式组织数据集
2. 修改 `datasets/data/data.yaml` 配置文件
3. 确保标注文件格式正确

### 训练配置
在 `train.py` 中调整训练参数：
- `epochs`: 训练轮数
- `batch`: 批处理大小
- `imgsz`: 输入图像尺寸
- `workers`: 数据加载线程数

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv11框架
- [Gradio](https://gradio.app/) - Web界面框架
- 北京林业大学 - 数据采集支持
- 半导体所 - 实地调研协助

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

- 📧 Email: GCameras77@Gmail.com/798957950@qq.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/MCRIS/issues)
- 📖 Wiki: [项目文档](https://github.com/your-username/MCRIS/wiki)

---

<div align="center">

**让城市更安全，让生活更美好** 🌟

*MCRIS - 智能守护每一个井盖*

</div>
