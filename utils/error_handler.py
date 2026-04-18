"""友好的错误处理和提示系统."""
from __future__ import annotations
from enum import Enum
from PyQt6.QtWidgets import QMessageBox, QWidget
from PyQt6.QtCore import Qt


class ErrorLevel(Enum):
    """错误级别."""
    INFO = "info"        # 信息提示，蓝色
    WARNING = "warning"  # 警告，黄色
    ERROR = "error"      # 错误，红色
    CRITICAL = "critical" # 严重错误，红色 + 阻止操作


# 错误消息模板
ERROR_TEMPLATES = {
    "midi_read": {
        "title": "无法读取 MIDI 文件",
        "message": "MIDI 文件可能已损坏或格式不正确。",
        "solution": (
            "• 尝试重新导出 MIDI 文件\n"
            "• 使用其他 MIDI 文件测试\n"
            "• 检查文件是否被其他程序占用"
        )
    },
    "audio_load": {
        "title": "无法加载音频",
        "message": "音频文件格式不支持或已损坏。",
        "solution": (
            "• 支持的格式：WAV, FLAC, MP3, OGG, AIFF\n"
            "• 推荐格式：44.1kHz 16-bit WAV\n"
            "• 尝试使用音频转换工具重新编码"
        )
    },
    "pitch_extract": {
        "title": "音高提取失败",
        "message": "无法分析音频的音高信息。",
        "solution": (
            "• 检查音频文件是否损坏\n"
            "• 确认音频采样率为 44.1kHz\n"
            "• 尝试重新导入音频文件\n"
            "• 重启应用程序"
        )
    },
    "render": {
        "title": "渲染错误",
        "message": "音频渲染过程中发生错误。",
        "solution": (
            "• 检查系统内存是否充足\n"
            "• 尝试减少音频长度\n"
            "• 关闭其他占用资源的程序"
        )
    },
    "file_save": {
        "title": "保存失败",
        "message": "无法保存项目文件。",
        "solution": (
            "• 检查磁盘空间是否充足\n"
            "• 确认目标文件夹有写入权限\n"
            "• 尝试保存到其他位置"
        )
    },
    "file_export": {
        "title": "导出失败",
        "message": "无法导出音频文件。",
        "solution": (
            "• 检查目标文件夹是否有写入权限\n"
            "• 确认文件名不包含非法字符\n"
            "• 尝试导出到其他位置"
        )
    },
    "model_load": {
        "title": "模型加载失败",
        "message": "无法加载 AI 模型文件。",
        "solution": (
            "• 检查模型文件是否存在\n"
            "• 重新下载模型文件\n"
            "• 联系技术支持获取帮助"
        )
    },
}


def show_error(parent: QWidget, error_type: str, details: str = "", 
               custom_message: str = None, custom_solution: str = None):
    """显示友好的错误提示。
    
    Args:
        parent: 父窗口
        error_type: 错误类型（使用预定义模板）
        details: 技术细节（可选）
        custom_message: 自定义消息（可选）
        custom_solution: 自定义解决方案（可选）
    """
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle(ERROR_TEMPLATES.get(error_type, {}).get("title", "错误"))
    
    # 构建消息
    message = custom_message or ERROR_TEMPLATES.get(error_type, {}).get("message", "发生错误。")
    solution = custom_solution or ERROR_TEMPLATES.get(error_type, {}).get("solution", "")
    
    full_message = f"{message}\n\n"
    if solution:
        full_message += f"💡 建议解决方案：\n{solution}\n\n"
    if details:
        full_message += f"📋 技术细节：\n{details}"
    
    msg.setText(full_message)
    
    # 添加标准按钮
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()


def show_friendly_exception(parent: QWidget, exception: Exception, 
                           context: str = ""):
    """根据异常类型显示友好的错误提示。
    
    Args:
        parent: 父窗口
        exception: 捕获的异常
        context: 异常发生的上下文
    """
    error_msg = str(exception)
    
    # 根据错误消息匹配错误类型
    error_type = "render"  # 默认
    
    if "MIDI" in error_msg or "midi" in error_msg:
        error_type = "midi_read"
    elif "audio" in error_msg.lower() or "wav" in error_msg.lower():
        error_type = "audio_load"
    elif "pitch" in error_msg.lower() or "f0" in error_msg.lower():
        error_type = "pitch_extract"
    elif "save" in error_msg.lower() or "write" in error_msg.lower():
        error_type = "file_save"
    elif "export" in error_msg.lower():
        error_type = "file_export"
    elif "model" in error_msg.lower() or "onnx" in error_msg.lower():
        error_type = "model_load"
    
    # 添加上下文信息
    full_details = error_msg
    if context:
        full_details = f"{context}\n\n{error_msg}"
    
    show_error(parent, error_type, details=full_details)


def confirm_action(parent: QWidget, title: str, message: str, 
                   detailed_text: str = "") -> bool:
    """显示确认对话框。
    
    Args:
        parent: 父窗口
        title: 对话框标题
        message: 确认消息
        detailed_text: 详细信息（可选）
    
    Returns:
        True 如果用户确认，False 否则
    """
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Question)
    msg.setWindowTitle(title)
    msg.setText(message)
    
    if detailed_text:
        msg.setDetailedText(detailed_text)
    
    msg.setStandardButtons(
        QMessageBox.StandardButton.Yes | 
        QMessageBox.StandardButton.No
    )
    msg.setDefaultButton(QMessageBox.StandardButton.No)
    
    return msg.exec() == QMessageBox.StandardButton.Yes


def show_info(parent: QWidget, title: str, message: str, 
              detailed_text: str = ""):
    """显示信息提示。
    
    Args:
        parent: 父窗口
        title: 标题
        message: 消息内容
        detailed_text: 详细信息（可选）
    """
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setWindowTitle(title)
    msg.setText(message)
    
    if detailed_text:
        msg.setDetailedText(detailed_text)
    
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()
