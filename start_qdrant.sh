#!/bin/bash
# Qdrant 服务启动脚本

set -e

# 配置
QDrant_PORT=${QDrant_PORT:-6333}
QDrant_HOST=${QDrant_HOST:-0.0.0.0}
QDrant_DATA_DIR=${QDrant_DATA_DIR:-./qdrant_data}
QDrant_LOG_FILE=${QDrant_LOG_FILE:-qdrant.log}

# 检查 Qdrant 二进制文件
QDrant_BIN=""
if [ -f "./qdrant" ]; then
    QDrant_BIN="./qdrant"
elif [ -f "$HOME/qdrant" ]; then
    QDrant_BIN="$HOME/qdrant"
elif command -v qdrant &> /dev/null; then
    QDrant_BIN="qdrant"
else
    echo "错误: 找不到 Qdrant 二进制文件"
    echo ""
    echo "请先下载 Qdrant:"
    echo "1. 访问 https://github.com/qdrant/qdrant/releases"
    echo "2. 下载适合你系统的版本（Linux x86_64）"
    echo "3. 解压后放到当前目录或 PATH 中"
    echo ""
    echo "或者使用 Docker:"
    echo "  docker run -p 6333:6333 -v \$(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant"
    exit 1
fi

# 创建数据目录
mkdir -p "$QDrant_DATA_DIR"

# 检查端口是否被占用
if lsof -Pi :$QDrant_PORT -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -tlnp 2>/dev/null | grep -q ":$QDrant_PORT " || ss -tlnp 2>/dev/null | grep -q ":$QDrant_PORT "; then
    echo "警告: 端口 $QDrant_PORT 已被占用"
    echo "如果 Qdrant 已经在运行，可以直接使用"
    echo "检查进程: ps aux | grep qdrant"
    exit 1
fi

# 启动 Qdrant
echo "启动 Qdrant 服务..."
echo "  端口: $QDrant_PORT"
echo "  数据目录: $QDrant_DATA_DIR"
echo "  日志文件: $QDrant_LOG_FILE"
echo ""

# 后台运行
nohup "$QDrant_BIN" \
    --config-path /dev/null \
    --uri "http://$QDrant_HOST:$QDrant_PORT" \
    --storage-path "$QDrant_DATA_DIR" \
    > "$QDrant_LOG_FILE" 2>&1 &

QDrant_PID=$!
echo "Qdrant 已启动，PID: $QDrant_PID"
echo "日志文件: $QDrant_LOG_FILE"
echo ""

# 等待服务启动
echo "等待服务启动..."
sleep 3

# 检查服务是否正常运行
if curl -s http://localhost:$QDrant_PORT/health > /dev/null 2>&1; then
    echo "✓ Qdrant 服务运行正常"
    echo "  访问地址: http://localhost:$QDrant_PORT"
    echo "  Web UI: http://localhost:$QDrant_PORT/dashboard"
    echo ""
    echo "停止服务: kill $QDrant_PID"
else
    echo "✗ Qdrant 服务启动失败，请查看日志: $QDrant_LOG_FILE"
    tail -20 "$QDrant_LOG_FILE"
    exit 1
fi











