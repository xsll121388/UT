"""检查HiFiGAN模型的详细信息"""
import onnxruntime as ort
import numpy as np

model_path = "models/hifigan.onnx"

print(f"加载模型: {model_path}")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("\n输入:")
for inp in session.get_inputs():
    print(f"  名称: {inp.name}")
    print(f"  形状: {inp.shape}")
    print(f"  类型: {inp.type}")
    print()

print("输出:")
for out in session.get_outputs():
    print(f"  名称: {out.name}")
    print(f"  形状: {out.shape}")
    print(f"  类型: {out.type}")
    print()

# 测试一个小输入
print("测试推理...")
n_frames = 10
mel_test = np.random.randn(1, n_frames, 128).astype(np.float32)
f0_test = np.ones((1, n_frames), dtype=np.float32) * 220.0

try:
    output = session.run(None, {"mel": mel_test, "f0": f0_test})
    print("Inference success")
    print(f"  Output shape: {output[0].shape}")
    print(f"  Output range: [{output[0].min():.3f}, {output[0].max():.3f}]")
except Exception as e:
    print(f"Inference failed: {e}")
