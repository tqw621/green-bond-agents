#!/bin/bash
# ============================================
# 绿色债券模拟系统 — 云服务器一键部署脚本
# 适用系统: Ubuntu 20.04+ / CentOS 7+
# 用法: chmod +x deploy.sh && ./deploy.sh
# ============================================

set -e

APP_DIR="/opt/green-bond-app"
echo "🌿 绿色债券模拟系统 — 部署脚本"
echo "========================================"

# ---- 0. 检查 Docker 环境 ----
if ! command -v docker &> /dev/null; then
    echo "📦 安装 Docker..."
    curl -fsSL https://get.docker.com | bash
    sudo systemctl enable docker && sudo systemctl start docker
    sudo usermod -aG docker "$USER"
    echo "⚠️  请重新登录以使 docker 用户组生效，然后重新运行本脚本"
    exit 0
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "📦 安装 docker-compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# ---- 1. 检查/创建目录 ----
if [ ! -d "$APP_DIR" ]; then
    echo "❌ 请先将项目代码上传到 $APP_DIR"
    echo "   git clone <repo-url> $APP_DIR"
    echo "   或 scp -r ./green-bond-agents user@server:$APP_DIR"
    exit 1
fi

cd "$APP_DIR"

# ---- 2. 检查 .env 配置 ----
if [ ! -f ".env" ]; then
    echo "📝 未找到 .env 文件，从 .env.example 创建..."
    cp .env.example .env
    echo "模型地址和名称可在 .env 中调整。API Key 由使用者在网页中输入。"
fi

# ---- 3. 创建数据目录 ----
mkdir -p "$APP_DIR/data"
chmod 755 "$APP_DIR/data"

# ---- 4. 可选：配置 Docker 国内镜像加速 ----
if [ ! -f "/etc/docker/daemon.json" ]; then
    echo "📦 配置 Docker 阿里云镜像加速..."
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<'DOCKER_EOF'
{
  "registry-mirrors": ["https://registry.cn-hangzhou.aliyuncs.com"]
}
DOCKER_EOF
    sudo systemctl restart docker
fi

# ---- 5. 构建并启动 ----
echo "🔨 构建并启动服务..."
docker-compose up -d --build

# ---- 6. 验证 ----
echo ""
echo "========================================"
echo "✅ 部署完成！"
echo ""
echo "检查服务状态："
docker-compose ps
echo ""
echo "查看日志："
echo "  docker-compose logs -f app"
echo ""
echo "访问地址："
echo "  http://$(curl -s ifconfig.me 2>/dev/null || echo '<服务器IP>')"
echo ""
echo "健康检查："
echo "  curl http://localhost:8504/_stcore/health"
