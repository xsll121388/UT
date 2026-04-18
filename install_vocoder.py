"""
从 .oudep 包中提取 ONNX 模型并替换当前的 hifigan.onnx
"""
import zipfile
import shutil
import os

OUDEP_PATH = os.path.join(os.path.dirname(__file__), "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
TARGET_PATH = os.path.join(MODELS_DIR, "hifigan.onnx")
BACKUP_PATH = os.path.join(MODELS_DIR, "hifigan.onnx.bak")

def main():
    if not os.path.exists(OUDEP_PATH):
        print(f"[错误] 找不到文件: {OUDEP_PATH}")
        return

    # 找到包内的 .onnx 文件
    with zipfile.ZipFile(OUDEP_PATH, "r") as zf:
        onnx_files = [f for f in zf.namelist() if f.endswith(".onnx")]
        if not onnx_files:
            print("[错误] .oudep 包内没有找到 .onnx 文件")
            return

        onnx_name = onnx_files[0]
        print(f"[信息] 找到模型: {onnx_name}")

        # 备份旧模型
        if os.path.exists(TARGET_PATH):
            shutil.copy2(TARGET_PATH, BACKUP_PATH)
            print(f"[信息] 已备份旧模型 → hifigan.onnx.bak")

        # 提取并重命名
        with zf.open(onnx_name) as src, open(TARGET_PATH, "wb") as dst:
            shutil.copyfileobj(src, dst)

    size_mb = os.path.getsize(TARGET_PATH) / 1024 / 1024
    print(f"[完成] 模型已替换: models/hifigan.onnx ({size_mb:.1f} MB)")
    print("[提示] 如需回滚，将 hifigan.onnx.bak 重命名为 hifigan.onnx 即可")

if __name__ == "__main__":
    main()
