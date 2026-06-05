---
comments: true
---

# ⑤ HTTP 请求：礼貌地向服务器"要东西"

TCP 连上了，TLS 加密了——终于可以说话了。浏览器用 **HTTP（HyperText Transfer Protocol）** 向服务器发送请求：**"请把首页给我"**。

!!! tip "核心要点"
    HTTP 是无状态、请求-响应模式的应用层协议。一个完整的 HTTP 事务包含 **请求报文** 和 **响应报文**。HTTP/1.1 是文本协议，HTTP/2 是二进制帧协议，HTTP/3 基于 QUIC（UDP）。

## 1. HTTP 请求报文解剖

一个典型的 HTTP 请求长这样：

```
GET /index.html HTTP/1.1                  ← 请求行
Host: www.example.com                     ← 请求头
User-Agent: Mozilla/5.0 (Windows NT 10.0)
Accept: text/html,application/xhtml+xml
Accept-Language: zh-CN,zh;q=0.9
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Cookie: session_id=abc123; theme=dark
                                          ← 空行（请求头结束标志）

(如果是 POST，这里会有请求体 Body)
```

### 三部分拆分

```
┌─────────────────────────────────────────┐
│  请求行    │ GET /index.html HTTP/1.1    │ ← 方法 + URI + 协议版本
├─────────────────────────────────────────┤
│  请求头    │ Host: ..., User-Agent: ...  │ ← 浏览器信息、Cookie等
├─────────────────────────────────────────┤
│  空行      │                            │ ← CRLF（\r\n）
├─────────────────────────────────────────┤
│  请求体    │ (POST/PUT 时才有)           │ ← 上传的数据
└─────────────────────────────────────────┘
```

## 2. HTTP 方法（动词）

| 方法 | 语义 | 幂等？ | 安全？ | 用途 |
|------|------|--------|--------|------|
| **GET** | 获取资源 | ✅ 是 | ✅ 是 | 访问网页、查数据 |
| **POST** | 创建资源 | ❌ 否 | ❌ 否 | 提交表单、上传文件 |
| **PUT** | 更新资源（全量替换） | ✅ 是 | ❌ 否 | REST API 更新 |
| **PATCH** | 更新资源（部分修改） | ❌ 否 | ❌ 否 | REST API 局部更新 |
| **DELETE** | 删除资源 | ✅ 是 | ❌ 否 | 删除数据 |
| **HEAD** | 只取响应头 | ✅ 是 | ✅ 是 | 检查资源是否存在 |
| **OPTIONS** | 查询支持的方法 | ✅ 是 | ✅ 是 | CORS 预检请求 |

> 幂等 = 执行 N 次和执行 1 次效果一样。安全 = 不修改服务器数据。

## 3. 关键请求头解读

| 请求头 | 含义 | 示例值 |
|--------|------|--------|
| `Host` | 目标主机（虚拟主机用） | `www.baidu.com` |
| `User-Agent` | 浏览器身份标识 | `Chrome/120.0...` |
| `Accept` | 期望的响应类型 | `text/html` |
| `Accept-Encoding` | 支持的压缩算法 | `gzip, deflate, br` |
| `Accept-Language` | 语言偏好 | `zh-CN,zh;q=0.9` |
| `Cookie` | 客户端存储的 Cookie | `session=xyz` |
| `Connection` | 连接管理 | `keep-alive` |
| `Referer` | 来源页面 URL | `https://www.google.com/` |
| `Cache-Control` | 缓存策略 | `no-cache` |

!!! tip "Accept-Encoding 的妙用"
    服务器看到 `Accept-Encoding: gzip, br` 后，会用 Brotli 或 Gzip 压缩 HTML 再返回。一个 200KB 的 HTML 压缩后可能只有 50KB，传输量直接砍掉 75%。

## 4. HTTP 响应报文

服务器的回复：

```
HTTP/1.1 200 OK                               ← 状态行
Content-Type: text/html; charset=UTF-8         ← 响应头
Content-Encoding: gzip
Cache-Control: max-age=3600
Set-Cookie: session=xyz; HttpOnly; Secure
Content-Length: 12345
                                               ← 空行
<!DOCTYPE html>                                ← 响应体
<html>
<head>...</head>
<body>...</body>
</html>
```

## 5. 状态码速查

| 范围 | 类别 | 常见状态码 |
|------|------|-----------|
| **1xx** | 信息 | `100 Continue`, `101 Switching Protocols` |
| **2xx** | 成功 | `200 OK`, `201 Created`, `204 No Content` |
| **3xx** | 重定向 | `301 永久`, `302 临时`, `304 Not Modified` |
| **4xx** | 客户端错误 | `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `429 Too Many Requests` |
| **5xx** | 服务器错误 | `500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable`, `504 Gateway Timeout` |

### 3xx 重定向的经典场景

```
浏览器请求 http://example.com
         │
         ├─ HTTP 301 → Location: https://example.com
         │   (永久重定向，浏览器会更新书签)
         │
         └─ 浏览器自动访问 https://example.com

浏览器请求 style.css (有缓存)
         │
         ├─ 请求头: If-None-Match: "abc123" (ETag)
         │
         └─ HTTP 304 Not Modified
             (文件没变，直接用缓存，不传 Body，省流量！)
```

## 6. HTTP 的版本演进

### HTTP/1.0 (1996)
- 每次请求新建一个 TCP 连接，用完就关
- 慢：每次都要三次握手

### HTTP/1.1 (1997)
- **持久连接 (Keep-Alive)**：复用一个 TCP 连接发多个请求
- **管线化 (Pipelining)**：不用等上一个响应就能发下一个请求（实际很少用，队头阻塞严重）
- **分块传输 (Chunked Transfer)**：不事先知道内容长度也能发

### HTTP/2 (2015)

**革命性的改进——二进制分帧**：

```
HTTP/1.1：文本协议，一个连接一次只能处理一个请求-响应
HTTP/2：  二进制帧，一个连接上多路复用 N 个请求
```

| 特性 | 说明 |
|------|------|
| **多路复用** | 一个 TCP 连接同时传输多个请求/响应 |
| **头部压缩 (HPACK)** | 请求头不再每次重复发送 |
| **服务器推送 (Server Push)** | 服务器主动推送 CSS/JS |
| **流优先级** | 重要资源优先传输 |
| **二进制协议** | 更紧凑，不再是可读文本 |

```
HTTP/1.1：请求A → 等待响应A → 请求B → 等待响应B
HTTP/2：  请求A ──── 响应A部分 ────
          请求B ──── 响应B部分 ────
          请求C ──── 响应C部分 ────  (交替传输，并行处理)
```

### HTTP/3 (2022)

直接抛弃 TCP，基于 **QUIC（基于 UDP）** ：

```
HTTP/1.1  === TCP
HTTP/2  ===== TCP (仍受 TCP 队头阻塞影响)
HTTP/3  ============= QUIC/UDP (0-RTT 握手，无队头阻塞)
```

!!! danger "HTTP/2 的队头阻塞"
    HTTP/2 多路复用了，但如果底层 TCP 丢了一个包，整个 TCP 连接的所有流都会被阻塞等待重传。HTTP/3 把"流"的概念下沉到 QUIC，每条流独立，一个丢包不影响其他。

## 7. 缓存的博弈

浏览器和服务器通过 HTTP 头协商缓存策略：

### 强缓存（不发请求）

| 响应头 | 含义 |
|--------|------|
| `Cache-Control: max-age=3600` | 缓存 3600 秒 |
| `Cache-Control: no-cache` | 每次验证后再用 |
| `Cache-Control: no-store` | 完全不缓存 |
| `Cache-Control: public` | CDN 也可缓存 |
| `Cache-Control: private` | 仅浏览器可缓存 |
| `Expires: Wed, 21 Oct 2025 07:28:00 GMT` | 过期时间（HTTP/1.0） |

### 协商缓存（发请求但可能不传 Body）

| 请求头 | 响应头 | 机制 |
|--------|--------|------|
| `If-None-Match` | `ETag` | 内容哈希比对 |
| `If-Modified-Since` | `Last-Modified` | 时间比对 |

```
浏览器缓存有 ETag="v1.0" 的 style.css
        │
        ├─ GET /style.css + If-None-Match: "v1.0" ──▶
        │
        ├─ ◀── 304 Not Modified (文件没变，不传 Body)
        │
        └─ 浏览器直接用缓存 ✅ (省了下载 200KB 的流量和时间)
```

## 小结

```
TLS 加密通道就绪
        │
        ├─ GET / HTTP/1.1
        │   Host: www.baidu.com
        │   User-Agent: Chrome/...
        │   Accept-Encoding: gzip, br
        │   Cookie: BAIDUID=...
        │
        ├─ (HTTP/2 的话，还会同时请求 CSS/JS/图片)
        │
        ├─ 数据经 TLS 加密 → TCP 分段 → IP 分包 → 链路层成帧
        │
        └─ 物理层电信号/光信号 → 经过路由器一跳一跳 → 服务器 → 下一阶段
```
