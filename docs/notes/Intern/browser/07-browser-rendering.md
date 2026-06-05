---
comments: true
---

# ⑦ 浏览器渲染：从 HTML 字符串到屏幕像素

服务器返回的 HTML 只是一串字符串。浏览器怎么把它变成你看到的彩色页面？答案是 **渲染流水线（Rendering Pipeline）**——一个精密的多阶段处理过程。

!!! tip "核心要点"
    渲染流水线分为 **解析 (Parse) → 样式 (Style) → 布局 (Layout) → 绘制 (Paint) → 合成 (Composite)** 五大阶段。每个阶段都可能成为性能瓶颈，理解它才能写出高性能页面。

## 1. 渲染流程全景图

```
HTML ──▶ DOM Tree ──┐
                     ├──▶ Render Tree ──▶ Layout ──▶ Paint ──▶ Composite ──▶ 屏幕
CSS  ──▶ CSSOM Tree ─┘
                         (样式计算)     (布局/重排)  (绘制)     (合成层)
```

> 在实际的 Chromium 引擎（Blink）中，Layout 和 Paint 不是全页面一次性完成的，而是先做的局部处理。

## 2. 第一步：解析 HTML → 构建 DOM 树

浏览器收到的是字节流，首先解码为字符，然后**词法分析**为 Token，最后构建 DOM 树。

### 字节 → 字符 → Token → 节点 → DOM 树

```
<html><body><p>Hello</p></body></html>
    │
    ├─ 网络线程收到字节流
    ├─ 解码为 Unicode 字符
    ├─ 词法分析 (HTMLTokenizer) → Token 流
    │    StartTag(html) → StartTag(body) → StartTag(p) → Character(Hello)
    │    → EndTag(p) → EndTag(body) → EndTag(html)
    ├─ DOM 构建器 (TreeBuilder) → DOM 节点树
    │
    ▼
        html
         │
        body
         │
         p
         │
      "Hello"
```

### 关键特性

**① 渐进式解析**：HTML 解析器不会等整个文档下载完才开始。收到第一个 chunk 就开始解析。

**② 容忍错误**：浏览器的 HTML 解析器非常宽容。即使 HTML 写得一塌糊涂（标签没闭合、嵌套错误），它也能"猜测"你的意图。

```html
<b><i>Hello</b> World   <!-- 标签错误嵌套 -->
```

浏览器会修复为：

```html
<b><i>Hello</i></b><i> World</i>
```

**③ 遇到 `<script>` 会阻塞**：没有 `async`/`defer` 的 `<script>` 会暂停 HTML 解析，先下载并执行 JS。

**④ 遇到 `<link>` 异步处理**：CSS 的下载不阻塞 HTML 解析，但会阻塞渲染（等 CSS 全部就绪才渲染）。

## 3. 第二步：解析 CSS → 构建 CSSOM 树

CSS 也要构建成树，因为样式是可以继承的。

```css
body { font-size: 16px; }
p { color: red; }
p span { display: none; }
```

构建的 CSSOM：

```
           CSSOM
        ┌─────────┐
        │  body   │ → font-size: 16px (所有后代继承)
        │  font:16 │
        └────┬────┘
             │
        ┌────┴────┐
        │    p    │ → color: red
        │  color:red│
        └────┬────┘
             │
        ┌────┴────────┐
        │    span     │ → display: none
        │ display:none│
        └─────────────┘
```

!!! danger "CSS 阻塞渲染"
    CSS 是**渲染阻塞资源**。页面必须等所有 CSS 加载并解析完毕后才能渲染，否则会出现"闪烁"（先看到无样式页面，再突然变好看）。这就是为什么 `<link>` 要放在 `<head>` 里、Critical CSS 要内联。

## 4. 第三步：合并 → 渲染树 (Render Tree)

把 DOM 树和 CSSOM 树合并，生成 **渲染树**（也叫 Layout Tree）。

```
DOM: html→body→p→"Hello"→span→"World"
CSSOM: body{font:16px}, p{color:red}, span{display:none}
                          │
              ┌───────────┴───────────┐
              │   合并 (Style Recalc)  │
              └───────────┬───────────┘
                          ▼
              渲染树: body(16px)
                        │
                       p(red,16px)
                        │
                     "Hello" (red,16px)
                     
              注意: <span> 被略过了 (display:none)
```

- `display: none` 的元素不进入渲染树
- `visibility: hidden` 的元素**会**进入渲染树（占位但不可见）
- 伪元素（`::before`, `::after`）也会出现在渲染树中

### 样式计算（Style Recalc）

浏览器为每个 DOM 节点计算**最终样式**（Computed Style）。考虑：
- 浏览器默认样式（User Agent Stylesheet）
- 外部 CSS 规则
- 内联样式
- 继承的样式
- **层叠 (Cascade)**：按优先级决定最终值

```
优先级计算 (Specificity):
    !important    > 最高
    内联 style=""  > 1000
    #id           > 0100
    .class        > 0010
    tag           > 0001
```

在 DevTools 的 Elements → Computed 面板可以看到最终样式。

## 5. 第四步：布局 (Layout / Reflow)

渲染树知道每个元素的样式（颜色、字体），但不知道在屏幕上的**位置和大小**。布局阶段做这件事。

### 盒模型（Box Model）

每个元素都是一个矩形盒子：

```
┌────────────────────────────┐
│         margin             │ ← 外边距
│   ┌────────────────────┐   │
│   │      border        │   │ ← 边框
│   │   ┌────────────┐   │   │
│   │   │  padding   │   │   │ ← 内边距
│   │   │ ┌────────┐ │   │   │
│   │   │ │ content│ │   │   │ ← 内容区
│   │   │ └────────┘ │   │   │
│   │   └────────────┘   │   │
│   └────────────────────┘   │
└────────────────────────────┘

盒子总宽度 = margin-left + border-left + padding-left 
           + content-width + padding-right + border-right + margin-right
```

### 布局流程

```
<html> 视口宽度 = 1920px
  └─ <body> 宽度 = 1920px
       └─ <div> display=block, 宽度 = 1920px
            └─ <p> 宽度 = 1920px, 高度 = 文字撑开的行数 × 行高
                 └─ 文字 "Hello World" x=0, y=0, w=..., h=20px
```

布局是一个**递归计算**的过程：父元素确定后，子元素的几何信息才能确定。

!!! danger "触发重排 (Reflow) 的操作"
    以下操作会触发 Layout，代价高昂：
    - 修改 DOM 结构（增删节点）
    - 修改会影响几何属性的样式（width/height/margin/padding/border/position/top/left...）
    - 读取某些属性（offsetWidth/offsetHeight/getBoundingClientRect()...）
    - 调整窗口大小
    - 修改字体

## 6. 第五步：绘制 (Paint)

布局确定了元素的几何位置，但还没变成像素。Paint 阶段生成**绘制指令列表**。

```
Paint 不是真正的"画像素"，而是生成一份"绘制清单"：

  1. 先画背景色 (白色, 矩形 0,0 → 1920×1080)
  2. 再画边框
  3. 再画文字 "Hello" (字体16px, 红色, 坐标 8,8)
  4. 再画图片 logo.png (缩放至 200×100, 坐标 8,30)
  ...
```

这个清单按**层叠顺序**排列（z-index、position 等决定谁先画谁后画）。

## 7. 第六步：合成 (Composite)

现代浏览器不会所有东西在一个层上画，而是分层绘制：

```
                    ┌────────────────┐
  合成器线程合并 →  │  Layer 1: 背景  │
                    ├────────────────┤
                    │  Layer 2: 文字  │
                    ├────────────────┤
                    │  Layer 3: 视频  │ → 独立层，GPU 加速
                    ├────────────────┤
                    │  Layer 4: 弹窗  │
                    └────────────────┘
                          │
                          ▼  GPU 合成 → 最终屏幕像素
```

### 哪些元素会创建独立合成层？

- 3D Transform：`transform: translateZ(0)`
- `<video>`, `<canvas>`, `<iframe>`
- `will-change: transform`
- CSS 动画/过渡中的 `transform` 和 `opacity`
- `position: fixed`

!!! tip "合成层的威力"
    合成层的最大优势：**只改一个层不会触发重排和重绘**。比如用 `transform: translateX(100px)` 移动一个元素，只改变该层的位置，其他层毫发无损，全程在 GPU 上完成，丝般顺滑。

## 8. 重排 → 重绘 → 合成的代价

| 操作 | 触发 Layout | 触发 Paint | 触发 Composite | 代价 |
|------|:----------:|:----------:|:--------------:|------|
| `width: 200px` | ✅ 是 | ✅ 是 | ✅ 是 | 💰💰💰 最贵 |
| `color: red` | ❌ 否 | ✅ 是 | ✅ 是 | 💰💰 |
| `transform: translateX(10px)` | ❌ 否 | ❌ 否 | ✅ 是 | 💰 最便宜 |
| `opacity: 0.5` | ❌ 否 | ❌ 否 | ✅ 是 | 💰 最便宜 |

> 这就是为什么做动画一定要用 `transform` 和 `opacity`——它们只触发合成，不需要重排重绘。

## 9. 关键渲染路径优化

这是前端性能优化的核心。关键三点：

```
1. 减少关键资源数量  → 内联 Critical CSS，延迟加载非关键 CSS/JS
2. 减少关键字节数    → 压缩 (gzip/brotli) + 压缩 HTML/CSS/JS
3. 缩短关键路径长度  → 减少请求链的 RTT 数量
```

**首屏渲染优化清单**：

| 策略 | 具体做法 |
|------|----------|
| **内联 Critical CSS** | 首屏需要的 CSS 写在 `<style>` 里 |
| **async / defer JS** | `<script defer>` 不阻塞 HTML 解析 |
| **资源预加载** | `<link rel="preload">` 提前告诉浏览器 |
| **DNS 预解析** | `<link rel="dns-prefetch">` 提前做 DNS |
| **减少 DOM 深度** | 扁平的 DOM 树布局更快 |
| **图片懒加载** | `loading="lazy"` 或者 Intersection Observer |

## 10. 渲染全流程时间线

在 Chrome DevTools → Performance 面板录制一次页面加载，你会看到：

```
时间轴 →
┌──────┬──────────┬──────────┬─────────┬────────┬───────────┐
│ 网络  │ Parse HTML│  Style   │ Layout  │ Paint  │Composite  │
│ 请求  │→DOM树     │ Recalc   │(重排)   │(重绘)  │(合成)     │
└──────┴──────────┴──────────┴─────────┴────────┴───────────┘
  ~200ms  ~50ms      ~15ms      ~30ms    ~20ms    ~5ms
```

## 小结

```
服务器返回 HTML
        │
        ├─ ① 解析 HTML ──▶ DOM 树 (渐进式，遇<script>阻塞)
        │
        ├─ ② 解析 CSS  ──▶ CSSOM 树 (CSS 完全加载才继续)
        │
        ├─ ③ 合并 ──▶ 渲染树 (DOM + CSSOM → Render Tree)
        │
        ├─ ④ Layout ──▶ 计算每个元素的几何位置和大小
        │
        ├─ ⑤ Paint  ──▶ 生成绘制指令列表
        │
        ├─ ⑥ Composite ──▶ 分层、GPU 合成 → 最终像素
        │
        └─ 🎉 你看到了网页！
```
