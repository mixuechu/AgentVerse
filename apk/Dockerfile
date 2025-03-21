FROM alpine:3.18

# 安装 abuild 和其他必要工具
RUN apk add --no-cache abuild alpine-sdk doas

# 配置 doas 允许 abuild-keygen 提升权限
RUN echo "permit nopass root" > /etc/doas.conf

# 设置工作目录
WORKDIR /build

# 复制 APK 包构建文件
COPY hello_world /build/hello_world

# 设置构建环境
RUN abuild-keygen -i -n

# 构建 APK 包
RUN cd hello_world && abuild -F checksum && abuild -F -r