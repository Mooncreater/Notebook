---
comments: true
---

# ④ TLS 握手：穿上加密外衣

TCP 连接已建立，但它是**裸奔**的——数据明文传输，任何人都能偷看和篡改。HTTPS 中的 **S** 就是 **TLS（Transport Layer Security）**，它在 TCP 之上建立一条加密隧道。

!!! tip "核心要点"
    TLS 握手的核心目标三个：**身份验证**（确认你是谁）、**密钥协商**（商量用什么密码）、**加密通道**（后续数据用对称加密）。TLS 1.2 需要 2 RTT，TLS 1.3 优化为 1 RTT。

## 1. TLS 在协议栈中的位置

```
┌──────────────────────────┐
│      应用层 (HTTP)        │ ← 明文数据在此
├──────────────────────────┤
│      TLS 层              │ ← 加密/解密在此
├──────────────────────────┤
│      传输层 (TCP)         │
├──────────────────────────┤
│      网络层 (IP)          │
└──────────────────────────┘
```

> TLS 工作在应用层和传输层之间，对 HTTP 来说是透明的——HTTP 还是那个 HTTP，只是经过 TLS 加密后再交给 TCP。

## 2. TLS 1.2 握手详解（4 步）

这是目前最广泛使用的版本，共 2 RTT：

```
客户端                                    服务器
  │                                        │
  │ ─── ① ClientHello ──────────────▶   │
  │   支持的加密套件、TLS版本、随机数1      │
  │                                        │
  │ ◀── ② ServerHello + 证书 ────────  │
  │   Certificate + ServerKeyExchange    │
  │   ServerHelloDone                     │
  │                                        │
  │ ─── ③ ClientKeyExchange ────────▶  │
  │   ChangeCipherSpec                    │
  │   Finished (加密验证)                  │
  │                                        │
  │ ◀── ④ ChangeCipherSpec+Finished ── │
  │                                        │
  │ ═══════ 加密通道建立，开始传 HTTP ═══════ │
```

### 逐步拆解

**① ClientHello** — 客户端说 "我想用这些方法加密"

客户端发送：
- 支持的 **TLS 版本**（1.2, 1.3）
- 支持的**加密套件**列表，如 `TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256`
- 一个**客户端随机数** (Client Random)

**② ServerHello + 证书** — 服务器说 "好，用这个方案，这是我的身份证"

服务器回复：
- 选定的加密套件
- **服务器随机数** (Server Random)
- **数字证书**（包含服务器公钥 + CA 签名）
- 如果需要，发送 **ServerKeyExchange**（ECDHE 的 DH 参数）

**③ ClientKeyExchange** — 客户端验证证书，生成密钥

客户端：
1. 验证证书链：用操作系统/浏览器内置的 CA 根证书逐级验证
2. 生成 **Pre-Master Secret**（用服务器公钥加密后发送）
3. 用 Client Random + Server Random + Pre-Master Secret → 计算 **Master Secret** → 推导出**会话密钥**
4. 发送 **ChangeCipherSpec + Finished**（加密的验证消息）

**④ 服务器也切换加密** — 服务器用同样算法算出相同的会话密钥，回复 Finished

```
双方各自持有：Client Random + Server Random + Pre-Master Secret
           │
           └─ PRF (伪随机函数) ──▶ Master Secret
                                      │
                                      └─▶ 对称密钥 (AES-GCM key + IV + HMAC key)
```

## 3. 证书链验证

为什么你从来没手动安装过百度/淘宝的证书，浏览器却信任它们？因为**证书链**：

```
操作系统内置的 CA 根证书 (如 DigiCert, Let''s Encrypt)
         │ 签名
         ▼
    中间 CA 证书
         │ 签名
         ▼
   baidu.com 的服务器证书
```

验证链路：
1. 用 CA 根证书的公钥验证中间 CA 证书的签名
2. 用中间 CA 证书的公钥验证服务器证书的签名
3. 检查证书是否在有效期内
4. 检查证书的域名（CN/SAN）是否匹配
5. 检查证书是否被吊销（CRL / OCSP）

!!! danger "如果证书验证失败"
    浏览器会显示 **"您的连接不是私密连接" (NET::ERR_CERT_AUTHORITY_INVALID)**，阻止你访问。这不是 bug，是安全特性。

## 4. 加密套件命名解读

```
TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
│   │     │    │    │   │   │    └─── 哈希算法 (HMAC-SHA256)
│   │     │    │    │   │   └─────── 加密模式 (GCM = Galois/Counter Mode, AEAD)
│   │     │    │    │   └─────────── 密钥长度 (128位)
│   │     │    │    └─────────────── 对称加密算法 (AES)
│   │     │    └──────────────────── 认证方式 (RSA 签名)
│   │     └───────────────────────── 密钥交换算法 (ECDHE: 椭圆曲线临时 Diffie-Hellman)
│   └─────────────────────────────── 是 TLS 加密套件标识
└─────────────────────────────────── 协议族
```

## 5. TLS 1.3 的改进

TLS 1.3（2018）做了大量简化，把握手从 2 RTT 降到 1 RTT：

| 对比 | TLS 1.2 | TLS 1.3 |
|------|---------|---------|
| 握手 RTT | 2 RTT | 1 RTT（首次）/ 0 RTT（恢复） |
| 加密套件数量 | 300+ 种组合 | 5 种（精简） |
| 密钥交换 | RSA 或 DH | 只有 DH（前向安全性） |
| 不安全的算法 | 允许 RC4、CBC 等 | 全部移除 |
| 证书加密 | 明文传输 | 握手后可加密 |

### TLS 1.3 的核心优化

```
TLS 1.2:  ClientHello → ServerHello+Cert → ClientKeyExchange → Finished
          (需要等服务器选好密钥交换算法再发证书)

TLS 1.3:  ClientHello(+KeyShare) → ServerHello(+KeyShare)+Cert+Finished → Finished
          (ClientHello 中直接附上 DH 公钥，服务器一步回复搞定)
```

因为客户端在 ClientHello 中直接提供了 DH 密钥共享参数，服务器可以立即计算出会话密钥。

## 6. Session Resumption：免重复握手

每次都做完整的 TLS 握手太慢了。TLS 提供了两种恢复机制：

| 机制 | 原理 | 效果 |
|------|------|------|
| **Session ID** | 服务器缓存会话参数，客户端下次带上 ID | 1 RTT |
| **Session Ticket** | 服务器把加密的会话参数发给客户端，下次客户端直接提交 | 1 RTT |
| **TLS 1.3 0-RTT** | 客户端用之前的 PSK 直接在第一个包发送加密数据 | 0 RTT |

> 这就是为什么你第二次访问一个网站比第一次快很多——不只是 HTML 缓存，TLS 握手也跳过了。

## 7. 实际观察：Chrome 开发者工具

打开 Chrome DevTools → Security 标签页，可以看到当前网站的 TLS 详细信息：

```
Connection: TLS 1.3
Cipher: TLS_AES_128_GCM_SHA256
Certificate: Valid (DigiCert)
```

打开 Timing 标签页，可以看到 TLS 握手耗时分解：

```
SSL (TLS) Negotiation: 45.2ms
```

## 小结

```
TCP 连接建立 (ESTABLISHED)
        │
        ├─ ① ClientHello  ──────────▶  (支持的加密方式 + 随机数1)
        │
        ├─ ◀────────── ServerHello + 证书 + 随机数2
        │      客户端验证证书链
        │
        ├─ ③ ClientKeyExchange ────▶  (Pre-Master Secret)
        │      双方各自计算：Master Secret → 会话密钥
        │
        ├─ ◀────────── Finished (加密的验证消息)
        │
        └─ 加密隧道就绪 → HTTP 可以安全传输 → 下一阶段
```
