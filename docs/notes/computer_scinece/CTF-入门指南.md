# CTF 从入门到上手 —— 基于 GitHub 顶级仓库总结

> 资料来源：[ctf-wiki](https://github.com/ctf-wiki/ctf-wiki) (9.4k★)、[awesome-ctf](https://github.com/apsdehal/awesome-ctf) (11.6k★)、[PayloadsAllTheThings](https://github.com/swisskyrepo/PayloadsAllTheThings) (78k★)、[HackTricks](https://github.com/HackTricks-wiki/hacktricks) (11.5k★)、[pwntools](https://github.com/Gallopsled/pwntools) (13.5k★)

---

## 一、CTF 是什么？

**CTF（Capture The Flag，夺旗赛）** 是网络安全领域的竞技比赛。起源于 1996 年的 **DEFCON** 黑客大会。

简单说：主办方在服务器/程序/网页里藏了"旗子"（flag，一段特定格式的字符串，如 `flag{th1s_1s_4_fl4g}`），参赛者要用各种技术手段找到它。

---

## 二、三种比赛形式

| 形式 | 怎么玩 | 特点 |
|------|--------|------|
| **Jeopardy 解题模式** | 像电视问答节目，题目分门别类，按难度给分 | 最主流，新手友好 |
| **Attack-Defense 攻防模式** | 每队有自己的服务器，攻别人的同时守自己的 | 更激烈，需要实时对抗 |
| **Mixed 混合模式** | 上面两种结合 | 综合难度最高 |

---

## 三、Jeopardy 模式的六大方向

绝大多数 CTF 比赛按以下六大方向出题：

```
┌──────────────────────────────────────────────────────┐
│  Web ──── 网站漏洞、SQL注入、XSS、反序列化           │
│  Pwn  ─── 二进制漏洞利用、栈溢出、堆利用、ROP          │
│  Reverse ─ 逆向工程、反编译、脱壳、算法分析            │
│  Crypto ── 密码学、古典密码、现代密码、哈希碰撞        │
│  Misc  ─── 隐写、取证、流量分析、编码转换              │
│  Mobile ── Android/iOS 逆向、应用安全                  │
└──────────────────────────────────────────────────────┘
```

---

## 四、每个方向一句话 + 难度 + 热门工具

### Web（最热门，入门首选）

**一句话**：找网站的安全漏洞，拿到不该拿到的数据。

| 难度 | 需要的基础 |
|------|-----------|
| ★★☆ 入门友好 | HTTP协议、PHP/Python/JS基础、数据库基础 |

**顶级攻击手法（PayloadsAllTheThings/HackTricks 总结）：**
- SQL 注入：在登录框输入 `' OR 1=1 --` 绕过验证
- XSS（跨站脚本）：把 `<script>alert(1)</script>` 注入页面
- CSRF：伪造请求，用你的身份偷偷转账
- SSRF：让服务器去访问不该访问的内网地址
- 文件上传漏洞：上传 `.php` 文件拿到服务器控制权
- 反序列化漏洞：操纵序列化数据执行任意代码

**必备工具**：Burp Suite（抓包改包）、浏览器 F12、sqlmap、dirsearch

---

### Pwn（最难最硬核）

**一句话**：给你一个程序，利用它的漏洞拿到 shell。

| 难度 | 需要的基础 |
|------|-----------|
| ★★★★★ 极难 | C语言、汇编、操作系统、内存管理 |

**核心攻击手法：**
- 栈溢出（Stack Overflow）：输入超长数据覆盖返回地址
- 堆利用（Heap Exploitation）：fastbin、tcache、unsorted bin 攻击
- ROP（Return-Oriented Programming）：没有可执行代码时，用程序自身的代码片段拼出攻击链
- 格式化字符串漏洞：`printf(user_input)` 没加 `%s`
- 整数溢出：`malloc(-1)` 的灾难后果

**必备工具**：[pwntools](https://github.com/Gallopsled/pwntools)（Python 库，CTF 神器）、[pwndbg](https://github.com/pwndbg/pwndbg)（GDB 插件）、IDA Pro / Ghidra、checksec



### Reverse（逆向工程）

**一句话**：给你一个编译好的程序（没有源代码），让你搞懂它怎么工作的，找到 flag。

| 难度 | 需要的基础 |
|------|-----------|
| ★★★★ 难 | C/C++、汇编、操作系统 |

**核心内容：**
- 静态分析：用 IDA Pro / Ghidra 看反编译代码，理解程序逻辑
- 动态调试：用 x64dbg / GDB 一步步跟程序，改寄存器、改内存
- 脱壳：对付加了壳的恶意软件（UPX、Themida、VMP）
- 算法逆向：程序把 flag 加密了，你要逆推出解密算法
- C++ 逆向：虚函数表、RTTI、异常处理——C++ 编译后很恶心

**必备工具**：IDA Pro（王者）/ Ghidra（免费替代）、x64dbg、OllyDbg、dnSpy（.NET）、jadx（Android 反编译）

**推荐入门仓库**：[x64dbg](https://github.com/x64dbg/x64dbg) (48k★)——Windows 上最好的开源调试器

---

### Crypto（密码学）

**一句话**：给你一些密文或者加密算法，找到漏洞解出明文。

| 难度 | 需要的基础 |
|------|-----------|
| ★★★ 中等 | 数学（数论、代数）、Python |

**核心内容：**
- 古典密码：凯撒、维吉尼亚、栅栏、培根密码（用我们的加密笔记就能解决！）
- 现代密码：RSA 攻击套路（共模攻击、低加密指数、Wiener 攻击）、AES 的各种误用（ECB 模式识别、CBC 字节翻转）
- 哈希：MD5/SHA1 碰撞、哈希长度扩展攻击
- 编码：Base64/32/16、Base91、Base92、UUencode——总有你没见过的编码

**必备工具**：[Ciphey](https://github.com/bee-san/Ciphey) (21k★)——自动识别并破解加密/编码，神器！CyberChef、Python（PyCryptodome）、SageMath、YAFU（大数分解）

---

### Misc（杂项，最不"杂"的杂项）

**一句话**：除上面四类之外的所有题，常考隐写、取证、流量分析。

| 难度 | 需要的基础 |
|------|-----------|
| ★~★★★ 不定 | 啥都可能涉及 |

**核心内容：**
- 隐写术（Steganography）：图片里藏文字、音频频谱里藏信息、LSB 最低比特位隐写
- 取证（Forensics）：给你一个内存镜像/硬盘镜像/数据包，还原作案过程
- 流量分析：Wireshark 分析 pcap 包，找到 HTTP 请求里的 flag、USB 键盘输入还原
- 编码转换：BaseXX 套娃、ppencode（只用 `( ) + [ ] !` 写的 JS 代码）、brainfuck
- 协议分析：Modbus、MQTT、蓝牙 BLE 数据包分析

**必备工具**：Wireshark、binwalk、foremost、steghide、zsteg、Volatility（内存取证）

---

### Mobile（移动安全）

**一句话**：给你一个 APK/IPA，逆向分析找漏洞。

| 难度 | 需要的基础 |
|------|-----------|
| ★★★★ 难 | Java/Kotlin、ARM汇编、Android/iOS 系统 |

**核心内容**：
- Android 逆向：APK → smali → Java 源码（jadx），找加密逻辑、网络请求
- iOS 逆向：IPA → Mach-O 分析
- 加固对抗：脱壳、反混淆、过反调试
- NDK 逆向：so 文件里的 native 代码（ARM 汇编）

**必备工具**：jadx、apktool、Frida、Objection、IDA Pro + ARM 插件

---



## 五、入门学习路线

来自 ctf-wiki 和众多 CTF 选手的共识：

```
第 1 步：选方向
  ├─ Web  ← 最简单，见效最快（推荐从这里开始！）
  ├─ Crypto ← 数学好选这个
  └─ Misc  ← 什么都想试试选这个

第 2 步：搭环境
  ├─ 装 Kali Linux / Parrot OS（安全专用系统，工具开箱即用）
  ├─ 或者 WSL2 + 手动装工具
  └─ 注册 CTF 在线平台

第 3 步：刷入门题
  ├─ BUUCTF（国内最大的 CTF 在线平台）
  ├─ NSSCTF（新兴平台，新手友好）
  ├─ CTFHub（按技能树分类，适合系统学习）
  └─ picoCTF（卡内基梅隆大学出品，全英文但极新手友好）

第 4 步：看 Writeup（解题报告）
  ├─ 每场比赛后都有人写 writeup
  ├─ ctf-wiki 上有历年真题解析
  └─ 看懂别人怎么做的，比自己做 10 题还有用

第 5 步：参加比赛
  ├─ 国内：网鼎杯、强网杯、高校 CTF 联赛
  ├─ 国际：DEFCON CTF（总决赛）、Google CTF、Plaid CTF
  └─ 关注 CTFtime.org 的比赛日历
```

---

## 六、GitHub 顶级资源速查表

### 学习类

| 仓库 | Stars | 一句话 |
|------|-------|--------|
| [ctf-wiki/ctf-wiki](https://github.com/ctf-wiki/ctf-wiki) | 9.4k | **中文首选**，六大方向全覆盖 |
| [firmianay/CTF-All-In-One](https://github.com/firmianay/CTF-All-In-One) | 4k+ | 另一本中文 CTF"百科全书" |
| [HackTricks-wiki/hacktricks](https://github.com/HackTricks-wiki/hacktricks) | 11.5k | 渗透测试/CTF 技巧大全 |
| [apsdehal/awesome-ctf](https://github.com/apsdehal/awesome-ctf) | 11.6k | CTF 工具/资源大合集 |

### 工具类

| 仓库 | Stars | 用途 |
|------|-------|------|
| [Gallopsled/pwntools](https://github.com/Gallopsled/pwntools) | 13.5k | **Pwn 必备**，Python 攻击框架 |
| [swisskyrepo/PayloadsAllTheThings](https://github.com/swisskyrepo/PayloadsAllTheThings) | 78k | **Web 攻击 payload 大全** |
| [pwndbg/pwndbg](https://github.com/pwndbg/pwndbg) | 10.5k | GDB 插件，调试神器 |
| [bee-san/Ciphey](https://github.com/bee-san/Ciphey) | 21.5k | 自动破解加密/编码 |
| [x64dbg/x64dbg](https://github.com/x64dbg/x64dbg) | 48.6k | Windows 开源调试器 |
| [juice-shop/juice-shop](https://github.com/juice-shop/juice-shop) | 13.3k | **Web 靶场**，故意有漏洞的商城网站供练习 |

### 在线平台

| 平台 | 特点 |
|------|------|
| [CTFtime.org](https://ctftime.org) | CTF 赛事日历 + 全球排名 |
| [BUUCTF](https://buuoj.cn) | 国内最大，历年真题超多 |
| [CTFHub](https://www.ctfhub.com) | 按技能树分类，新手友好 |
| [picoCTF](https://picoctf.org) | 全英文但题目质量极高 |
| [Attack-Defense](https://attackdefense.com) | 在线实验室，带 Kali 环境 |
| [HackTheBox](https://www.hackthebox.com) | 真实渗透测试环境 |

---

## 七、这个方向和我们之前的加密笔记有什么关系？

你刚学完的加密笔记（凯撒密码 → Base64 → AES → RSA → HTTPS）是 CTF Crypto 方向的基础。

**CTF Crypto 题目本质上就是：出题人故意用错了加密算法（或用了一些漏洞），你要找到这个漏洞然后解密。**

比如学过的：
- **凯撒密码** → CTF 里出现频率极高，经常和其他编码套娃
- **Base64** → CTF 里最常见的编码，经常套十几层（Base64→Base32→Base16→Base85...）
- **AES-ECB 模式** → 前面笔记说"千万别用 ECB"，CTF 就爱出 ECB 模式的漏洞题
- **RSA 低加密指数** → `e=3` 且明文太短 → 直接开三次方就能解密

---

## 🎯 总结

| 你想... | 从这里开始 |
|---------|-----------|
| 最快上手 | Web 方向 + ctf-wiki + BUUCTF 刷题 |
| 硬核挑战 | Pwn 方向 + pwntools + 刷堆题 |
| 用数学能力 | Crypto 方向 + Ciphey 工具 + 前面加密笔记 |
| 当瑞士军刀 | Misc 方向 + Wireshark + 各种隐写工具 |
| 想参加比赛 | CTFtime.org 找最近的比赛 + 组队 |

---

> **CTF 是学安全最快的方式——每一道题都是一个真实漏洞的简化版。**
> 看完这篇，打开 [ctf-wiki.org](https://ctf-wiki.org)，挑一个方向开始吧。
