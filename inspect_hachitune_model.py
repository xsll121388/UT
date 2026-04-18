"""检查HachiTune模型的输入输出格式"""
import onnxruntime as ort

model_path = "models/hifigan_hachitune.onnx"

print(f"Model: {model_path}")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("\nInputs:")
for inp in session.get_inputs():
    print(f"  Name: {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type: {inp.type}")
    print()

print("Outputs:")
for out in session.get_outputs():
    print(f"  Name: {out.name}")
    print(f"  Shape: {out.shape}")
    print(f"  Type: {out.type}")
    print()
