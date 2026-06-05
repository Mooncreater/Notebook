---
comments: true
---

# ② DNS 查询：把域名"翻译"成 IP 地址

浏览器拿到了域名 `www.baidu.com`，但它无法直接通过域名找到服务器——互联网上的每一台机器都用 **IP 地址** 定位。DNS（Domain Name System）就是互联网的**电话本**：输入域名，返回 IP。

!!! tip "核心要点"
    DNS 查询是一个 **逐级向上、层层缓存** 的过程。浏览器 → 操作系统 → 路由器 → ISP → 根服务器 → 顶级域 → 权威服务器，每一步都可能命中缓存。首次查询可能 200ms，缓存命中只需 0ms。

## 1. DNS 的缓存层级体系

在你真正发起网络 DNS 查询之前，系统已经做了多层缓存检查：

```
┌─────────────────────────────────────┐
│ ① 浏览器 DNS 缓存 (chrome://net-internals/#dns) │ ← 命中：0ms
├─────────────────────────────────────┤
│ ② 操作系统 DNS 缓存 (ipconfig /displaydns)     │ ← 命中：~1ms
├─────────────────────────────────────┤
│ ③ 本地 hosts 文件 (C:\Windows\System32\drivers\etc\hosts) │
├─────────────────────────────────────┤
│ ④ 路由器缓存                         │ ← 命中：~5ms
├─────────────────────────────────────┤
│ ⑤ ISP DNS 服务器 (如 114.114.114.114) │ ← 命中：~10-50ms
├─────────────────────────────────────┤
│ ⑥ 递归查询：根 → 顶级域 → 权威       │ ← 无缓存：100-200ms
└─────────────────────────────────────┘
```

### 各级缓存详解

**① 浏览器 DNS 缓存**：Chrome 默认缓存 DNS 记录约 1 分钟。你可以在 `chrome://net-internals/#dns` 看到所有缓存的域名。

**② 操作系统 DNS 缓存**：Windows 用 `ipconfig /displaydns` 查看，Linux 一般用 `systemd-resolved` 或 `nscd`。

**③ hosts 文件**：这是本地手动配置的域名映射，优先级极高。常用于开发环境（如 `127.0.0.1 myapp.local`）。

!!! danger "注意"
    hosts 文件可以被恶意软件修改来实现 DNS 劫持，比如把 `www.taobao.com` 指向钓鱼网站的 IP。

**④ 路由器缓存**：家用路由器通常自带 DNS 缓存功能，减轻上游压力。

**⑤ ISP DNS 服务器**：电信/联通/移动各自的 DNS 服务器，响应快但可能有劫持（比如输错域名时跳转到广告页）。

## 2. 递归查询全过程（无缓存时）

如果以上缓存全部未命中，DNS 解析器会发起递归查询：

```
你的电脑                    本地 DNS 解析器                   DNS 层级系统
   │                            │                               │
   │ ① 查询 www.baidu.com       │                               │
   │──────────────────────────▶│                               │
   │                            │ ② 问根服务器 "."              │
   │                            │──────────────────────────────▶│ 根 DNS
   │                            │ ◀─ "我不知道，去问 .com"      │
   │                            │                               │
   │                            │ ③ 问 .com 顶级域服务器        │
   │                            │──────────────────────────────▶│ .com TLD
   │                            │ ◀─ "我不知道，去问 baidu.com" │
   │                            │                               │
   │                            │ ④ 问 baidu.com 权威服务器      │
   │                            │──────────────────────────────▶│ baidu.com 权威
   │                            │ ◀─ "www.baidu.com = 110.242.68.66" │
   │                            │                               │
   │ ◀─ 返回 IP                 │                               │
   │                            │                               │
```

### 三步走理解

| 步骤 | 问谁 | 得到什么 |
|------|------|----------|
| 根 DNS 服务器 | 全球 13 组根服务器 | "`.com` 的 NS 服务器地址是 x.x.x.x" |
| 顶级域 (TLD) 服务器 | .com 的权威服务器 | "`baidu.com` 的 NS 服务器地址是 y.y.y.y" |
| 权威 DNS 服务器 | baidu.com 自己维护的 DNS | "`www.baidu.com` 的 A 记录 = `110.242.68.66`" |

!!! tip "全球 13 组根服务器"
    注意是 13 **组**，不是 13 台！实际上通过 Anycast 技术，全球有上千个根服务器镜像节点。中国境内就有多个根镜像（北京、上海等）。

## 3. DNS 记录类型

DNS 不仅仅存储 IP 地址，它有多种记录类型：

| 类型 | 含义 | 示例 |
|------|------|------|
| **A** | IPv4 地址 | `www.baidu.com → 110.242.68.66` |
| **AAAA** | IPv6 地址 | `www.baidu.com → 240e:e9:6000::1` |
| **CNAME** | 别名/规范名 | `www.baidu.com → www.a.shifen.com` |
| **MX** | 邮件服务器 | `baidu.com → mx.baidu.com` |
| **NS** | 权威 DNS 服务器 | `baidu.com → ns1.baidu.com` |
| **TXT** | 文本记录（SPF/DKIM 验证） | `"v=spf1 include:baidu.com ~all"` |
| **SRV** | 服务定位 | 指定特定服务的端口和主机 |

!!! tip "CNAME 的妙用"
    CDN 的核心原理之一：`www.example.com → CNAME → xxx.cdn.com → A → CDN 边缘节点 IP`。域名指向 CDN，CDN 再智能分配到最近的边缘节点。

## 4. 实际抓包：`nslookup` 调试

你可以亲自验证 DNS 查询过程：

```bash
# 用特定 DNS 服务器查询
nslookup www.baidu.com 8.8.8.8

# 输出示例：
# Server:  dns.google
# Address:  8.8.8.8
#
# Non-authoritative answer:
# Name:    www.a.shifen.com          ← CNAME 别名！
# Addresses:  110.242.68.66          ← 最终的 IPv4
#             240e:ff:e020:966::32   ← 最终的 IPv6
# Aliases:  www.baidu.com
```

可以看到 Baidu 实际用的是 `www.a.shifen.com` 这个 CDN 域名，通过 CNAME 别名指向。

## 5. DNS 安全问题与进化

### 传统 DNS 的问题

| 问题 | 说明 |
|------|------|
| **明文传输** | UDP 53 端口，中间人可以偷看/篡改 |
| **无身份验证** | 无法确认返回结果的真实性 |
| **DNS 劫持** | ISP/路由器可篡改响应（如插广告） |
| **DNS 缓存投毒** | 污染 DNS 服务器的缓存 |

### DNS over HTTPS (DoH) / DNS over TLS (DoT)

现代方案：把 DNS 查询加密进 HTTPS 或 TLS 隧道。

```
传统 DNS：UDP 53 端口，明文，谁都能看
DoH    ：HTTPS 443 端口，加密，混在普通网页流量中
DoT    ：TCP 853 端口，TLS 加密，独立的加密通道
```

Chrome 默认支持 DoH，在设置中可以开启。国内常见 DoH 服务：阿里 `dns.alidns.com`、DNSPod `doh.pub`。

## 6. 总结：DNS 查询的执行流程

```
浏览器拿到域名 "www.baidu.com"
        │
        ├─ ① 查浏览器 DNS 缓存 ── 命中？ → 返回 IP ✅
        │      └─ 未命中 ↓
        ├─ ② 调系统 gethostbyname() ── OS 缓存命中？ → 返回 IP ✅
        │      └─ 未命中 ↓
        ├─ ③ 查 hosts 文件 ── 有配置？ → 返回 IP ✅
        │      └─ 未命中 ↓
        ├─ ④ 路由 DNS 转发到 ISP DNS ── ISP 缓存命中？ → 返回 IP ✅
        │      └─ 未命中 ↓
        ├─ ⑤ 递归查询：根 → .com → baidu.com 权威
        │
        └─ 得到 IP：110.242.68.66 → 传给 TCP 层 → 下一阶段
```

> 现在浏览器知道了服务器的 IP 地址，接下来需要和它建立连接。这就是下一篇章的主角：TCP。
