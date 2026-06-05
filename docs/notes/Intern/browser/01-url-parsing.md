---
comments: true
---

# ① URL 解析：浏览器做的第一件事

你输入 `www.baidu.com` 按下 Enter，浏览器第一反应不是上网，而是：**"这家伙到底想干啥？"** 它需要判断这是一个网址，还是一个搜索词。

!!! tip "核心要点"
    URL 解析的本质是 **"输入的到底是什么"** 的判断 + **"URL 标准化"** 的处理。这一阶段耗时 < 1ms，完全在浏览器内完成，不涉及任何网络请求。

## 1. 输入判断：搜索 vs 网址

浏览器收到地址栏输入后，先判断它是网址还是搜索关键词：

| 输入 | 浏览器判断 | 实际行为 |
|------|-----------|----------|
| `baidu` | 不包含 `.` 或不完整 | 跳转搜索引擎搜索 "baidu" |
| `baidu.com` | 看起来像域名 | 自动补全为 `http://baidu.com` |
| `https://baidu.com` | 完整的 URL | 直接访问 |
| `192.168.1.1:8080` | IP 地址 + 端口 | 直接访问 |
| `localhost` | 保留关键字 | 访问本地服务 |

### 判断规则（Chrome 为例）

```
输入字符串
    ├── 包含空格 或 不以 "."/"://" 开头？
    │       ├── 是 → 搜索
    │       └── 否 → 继续
    ├── 是合法 URL scheme？（http/https/ftp/file...）
    │       └── 是 → 直接导航
    └── 看起来像域名？
            ├── 是 → 补全为 http(s):// 然后导航
            └── 否 → 搜索
```

> 这也是为什么输入 `taobao` 会去搜索，而输入 `taobao.com` 会直接访问。

## 2. URL 的完整结构

一个完整的 URL 各部分如下：

```
https://user:pass@www.example.com:443/path/to/page?key=val#fragment
└─┬─┘ └──┬──┘ └────┬────┘ └──┬──┘ └─────┬─────┘ └──┬──┘ └──┬───┘
 scheme  user:pass    host     port     path      query  fragment
 (协议)  (认证信息)  (主机)   (端口)   (路径)   (查询参数) (锚点)
```

| 部分 | 必需？ | 示例 | 说明 |
|------|--------|------|------|
| **scheme** | ✅ | `https` | 协议，浏览器支持 http/https/ftp/file/data/blob 等 |
| **host** | ✅ | `www.baidu.com` | 域名或 IP 地址 |
| **port** | ❌ | `443` | 省略时用默认值（http=80, https=443） |
| **path** | ❌ | `/s?wd=hello` | 资源路径 |
| **query** | ❌ | `?q=你好&page=2` | 键值对参数 |
| **fragment** | ❌ | `#section1` | 锚点，不会发送到服务器 |

!!! tip "user:pass 认证"
    `https://admin:123456@example.com` 这种写法技术上支持，但出于安全原因，现代浏览器大多废弃了对 URL 中嵌入用户名密码的支持。

## 3. URL 编码（Percent-Encoding）

URL 只允许 ASCII 字符集中的特定字符。那中文怎么办？答案是 **百分号编码**。

### 哪些字符需要编码？

- **保留字**：`:` `/` `?` `#` `[` `]` `@` `!` `$` `&` `'` `(` `)` `*` `+` `,` `;` `=`
- **不安全字符**：空格、`"` `<` `>` `%` `{` `}` `|` `\` `^` `~` `` ` ``
- **非 ASCII 字符**：中文、日文、emoji 等

### 编码规则

```
字符 → UTF-8 字节序列 → 每个字节前加 % → 完成
```

**例子**：

| 字符 | UTF-8 编码 → | 结果 |
|------|-------------|------|
| 空格 | 0x20 | `%20`（或 `+`） |
| `中` | E4 B8 AD | `%E4%B8%AD` |
| `!` | 0x21 | `%21` |

> 所以你在地址栏看到的中文 "你好"，背后传输的是 `%E4%BD%A0%E5%A5%BD`。

## 4. HSTS：强制 HTTPS

如果你输入 `baidu.com`，浏览器默认会先尝试 HTTP（端口 80）。但如果这个网站配置了 **HSTS（HTTP Strict Transport Security）**，浏览器会记住"这个域名必须用 HTTPS"。

### HSTS 工作流程

```
第一次访问：https://example.com → 响应头：Strict-Transport-Security: max-age=31536000
之后访问：
    http://example.com → 浏览器内部自动 307 重定向 → https://example.com
```

- 307 Internal Redirect：浏览器内部处理，不发请求，杜绝中间人攻击窗口
- `max-age=31536000`：有效期 1 年

!!! danger "安全要点"
    没有 HSTS 时，`http://xxx → 301 → https://xxx` 这个过程 **第一次请求是明文的**，可能被中间人拦截（SSL Stripping 攻击）。HSTS 消除了这个漏洞。

### HSTS Preload List

Chrome 内置了一个硬编码的 HSTS 列表，包含 `google.com`、`github.com` 等域名。即使你从未访问过这些网站，浏览器也会强制使用 HTTPS。

## 5. 安全检查：恶意网站拦截

在真正发起网络请求之前，浏览器还会做最后一件事：

1. 查本地 **Safe Browsing 黑名单**（Google/Edge/Firefox 各自维护）
2. 如果匹配 → 显示红色警告页面，阻止访问

这个检查基于 URL 哈希，不是完整 URL，隐私相对安全。

## 小结

```
用户输入 "www.baidu.com"
        │
        ├─ 判断：是网址（包含 "baidu.com"）
        │
        ├─ 补全：→ http://www.baidu.com/
        │
        ├─ 查 HSTS：baidu.com 在 HSTS 列表？
        │     是 → 升级为 https://www.baidu.com/
        │
        ├─ URL 编码：路径/参数如有中文则 Percent-Encode
        │
        ├─ 安全检查：Safe Browsing 黑名单？
        │
        └─ 提取 host → 传给 DNS 解析器 → 下一阶段
```
