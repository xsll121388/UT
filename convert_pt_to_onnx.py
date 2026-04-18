"""
将 PyTorch RMVPE 模型 (.pt) 转换为 ONNX 格式
使用方法: python convert_pt_to_onnx.py --input models/rmvpe.pt --output models/rmvpe.onnx
"""
import argparse
import os
import sys
import numpy as np

def convert_model(input_path: str, output_path: str):
    """Convert PyTorch model to ONNX format."""
    try:
        import torch
    except ImportError:
        print("错误: 需要安装 PyTorch")
        print("安装命令: pip install torch")
        sys.exit(1)
    
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)
    
    print(f"正在加载模型: {input_path}")
    
    # 尝试加载模型
    try:
        # 方法1: 尝试加载为 TorchScript
        model = torch.jit.load(input_path, map_location="cpu")
        model.eval()
        print("成功加载为 TorchScript 模型")
        
        # 转换为 ONNX
        dummy_input = torch.randn(1, 16000)  # 1秒音频 @ 16kHz
        dummy_threshold = torch.tensor(0.03, dtype=torch.float32)
        
        torch.onnx.export(
            model,
            (dummy_input, dummy_threshold),
            output_path,
            input_names=["waveform", "threshold"],
            output_names=["f0", "uv"],
            dynamic_axes={
                "waveform": {0: "batch_size", 1: "time"},
                "f0": {0: "batch_size", 1: "frames"},
                "uv": {0: "batch_size", 1: "frames"},
            },
            opset_version=11,
        )
        
    except Exception as e1:
        print(f"TorchScript 加载失败: {e1}")
        print("\n尝试加载为普通 PyTorch 模型...")
        
        try:
            # 方法2: 加载为普通 PyTorch 模型
            checkpoint = torch.load(input_path, map_location="cpu")
            
            if isinstance(checkpoint, dict):
                print("检测到 state_dict 格式")
                print("注意: 需要模型类定义才能正确加载")
                print("\n请提供模型类定义，或使用以下方法:")
                print("1. 使用原始代码中的模型类加载并重新保存")
                print("2. 或使用 torch.jit.script/trace 转换为 TorchScript 后再转换")
                sys.exit(1)
            else:
                # 是一个完整的模型
                model = checkpoint
                model.eval()
                print("成功加载为普通 PyTorch 模型")
                
                # 转换为 ONNX
                dummy_input = torch.randn(1, 16000)
                dummy_threshold = torch.tensor(0.03, dtype=torch.float32)
                
                torch.onnx.export(
                    model,
                    (dummy_input, dummy_threshold),
                    output_path,
                    input_names=["waveform", "threshold"],
                    output_names=["f0", "uv"],
                    dynamic_axes={
                        "waveform": {0: "batch_size", 1: "time"},
                        "f0": {0: "batch_size", 1: "frames"},
                        "uv": {0: "batch_size", 1: "frames"},
                    },
                    opset_version=11,
                )
        
        except Exception as e2:
            print(f"转换失败: {e2}")
            sys.exit(1)
    
    print(f"\n成功! ONNX 模型已保存到: {output_path}")
    
    # 验证模型
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        print(f"模型验证成功!")
        print(f"输入: {[inp.name for inp in sess.get_inputs()]}")
        print(f"输出: {[out.name for out in sess.get_outputs()]}")
    except Exception as e:
        print(f"模型验证警告: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 PyTorch RMVPE 模型转换为 ONNX")
    parser.add_argument("--input", "-i", default="models/rmvpe.pt", help="输入的 .pt 文件路径")
    parser.add_argument("--output", "-o", default="models/rmvpe.onnx", help="输出的 .onnx 文件路径")
    
    args = parser.parse_args()
    convert_model(args.input, args.output)
