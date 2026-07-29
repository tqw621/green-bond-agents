"""
工具模块 - 占位
后续可添加：数据加载、格式转换、可视化辅助函数等
"""

def format_rate(rate: float) -> str:
    """格式化百分比"""
    return f"{rate:.2%}"


def format_amount(amount: float) -> str:
    """格式化金额（亿元）"""
    return f"{amount:.2f}亿元"
