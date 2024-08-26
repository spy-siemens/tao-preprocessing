import onnxruntime as ort
import image_process

WIDTH = 224
HEIGHT = 224


# 加载 ONNX 模型
session = ort.InferenceSession('./model.onnx')

# 读取图像
image = image_process.generate_input("./simatic_photos/S7_1200/IMG_1263.jpg")

# 推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image})

print(outputs)


