---
comments : true
---

# Bonus1 — 透明物体绘制（Transparency）

!!! tip "核心要点"
    透明渲染的核心矛盾：z-buffer 只存最近深度，但透明物体需要看到后面的面。
    三种方案 = Alpha Testing（一刀切）→ Alpha Blending（半透明但依赖顺序）→ Depth Peeling（顺序无关，最正确但最慢）。

## 0. 前置知识：你为什么需要这篇笔记

如果你刚接触 OpenGL，先确保你理解这些概念：
- **z-buffer（深度缓冲）**：GPU 自动记录每个像素的深度。画新片元时，深度 ≥ z-buffer 直接丢弃。
- **Blending（混合）**：新片元和屏幕上已有颜色按方程混合，由 `glBlendFunc` 控制。
- **Framebuffer（帧缓冲）**：渲染目标不仅可以是屏幕，也可以是自己创建的内存纹理。

这三个概念是本 bonus 的基石。

## 1. 问题：为什么透明物体这么难画？

### 1.1 不透明物体的渲染逻辑

```
画三角形A（近）→ z-buffer 记录深度
画三角形B（远）→ z-buffer 说"这里已经有更近的片元了"→ 丢弃
```

不透明物体：z-buffer = 完美解决方案。

### 1.2 透明物体的困境

透明物体如果也按这个逻辑，后面的面被 z-buffer 扔掉了，混合就无法进行：

```
画透明前面 → z-buffer = 0.3
画透明后面 → z-buffer 说"深度 0.5 > 0.3，丢弃"→ 后面的颜色没了！
```

**本质矛盾**：z-buffer 的"只留最近的"与透明的"后面的也得画"互斥。

### 1.3 三种方案的定位

| 方案 | 核心思路 | 正确性 | 性能 | 适合场景 |
|------|----------|--------|------|----------|
| Alpha Testing | 绕开混合，用 discard 一刀切 | 差（无半透明） | 最快 | 树叶、栅栏、贴花 |
| Alpha Blending | 两 Pass：先建深度再混合 | 中等（缺背面） | 快 | 粒子、简单半透物体 |
| Depth Peeling | 逐层剥离，从前往后混合 | 完全正确 | 慢 | 高质量透明渲染 |

---

## 2. Alpha Testing（透明度测试）

### 2.1 核心原理

每个片元看一眼纹理的 alpha 值：
- alpha < 阈值（如 0.5）→ `discard`，等价于这个片元从未存在
- alpha ≥ 阈值 → 正常画（完全不透明）

**没有中间态**，不存在"30% 透明"这种概念。

### 2.2 代码实现

片元着色器 `alpha_test.frag`：

```glsl
void main() {
    vec4 texColor = texture(transparentTexture, fTexCoord);
    
    // ★ 核心：alpha 不足就直接丢弃
    if (texColor.a < 0.5f) {
        discard;
    }

    // ... Lambert 光照计算（和不透明物体一样）...
    vec3 ambient = material.ka * material.albedo;
    vec3 diffuse = material.kd * texColor.rgb * max(dot(lightDir, normal), 0.0f);
    color = vec4(ambient + diffuse, material.transparent);
}
```

### 2.3 `discard` 的本质

`discard` 是 GLSL 关键词，执行后：
- 该片元不写入颜色缓冲
- 该片元不写入深度缓冲
- 该片元不触发混合

等价于：这个片元在管线中直接被抹掉了，仿佛这个三角形在这里没有覆盖这个像素。

### 2.4 CPU 端

完全不需要特殊 OpenGL 状态——就是普通的 shader 使用 + 传 uniform + `glDrawElements`：

```cpp
_alphaTestingShader->use();
// 设 MVP、光照、材质 uniform...
_transparentTexture->bind(0);
_knot->draw();  // 就是普通绘制
```

### 2.5 优缺点

**优点**：不需要排序，不需要混合，极快。z-buffer 正常工作。

**缺点**：锯齿严重。想象边缘像素 alpha = 0.49 → discard，相邻像素 alpha = 0.51 → 画出。一条硬边。

---

## 3. Alpha Blending（透明度混合）

### 3.1 核心原理

使用 OpenGL 的混合管线，把新片元按透明度叠在已有颜色上。

但直接开混合画有问题：knot 自己的不同面绘制顺序不可控，可能导致错误遮挡。

### 3.2 两 Pass 策略

```
Pass 1 — 只建深度（和画不透明物体一样）
  glColorMask(FALSE, FALSE, FALSE, FALSE)  // RGBA 全关
  glDepthMask(TRUE)                        // 深度写入开
  _knot->draw()                            // z-buffer 现在有 knot 的深度信息

Pass 2 — 只画颜色 + 混合
  glColorMask(TRUE, TRUE, TRUE, TRUE)      // 恢复颜色
  glDepthMask(FALSE)                       // 不写深度
  glDepthFunc(GL_LEQUAL)                   // ★ 等深度也通过
  glEnable(GL_BLEND)
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
  _knot->draw()

恢复状态
  glDepthFunc(GL_LESS)
  glDepthMask(TRUE)
  glDisable(GL_BLEND)
```

### 3.3 为什么必须 `GL_LEQUAL`？

这是初学最容易踩的坑：

- Pass 1 和 Pass 2 画的是完全相同的几何体
- Pass 1 的深度已经写入了 z-buffer，值是 0.3
- Pass 2 的同一个片元深度也是 0.3
- 默认 `GL_LESS`：要求 0.3 < 0.3 → false → 丢弃
- 改成 `GL_LEQUAL`：要求 0.3 ≤ 0.3 → true → 通过

!!! warning "常见 bug"
    如果 Alpha Blending 模式下什么也看不到，99% 是忘了 `glDepthFunc(GL_LEQUAL)`。

### 3.4 混合方程

```glsl
// alpha_blend.frag
color = vec4(ambient + diffuse, material.transparent);
```

GPU 混合方程（`GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA`）：

$$C_{final} = C_{src} \cdot \alpha + C_{dst} \cdot (1 - \alpha)$$

- α = 0.8（80% 不透明）：最终色 = 80%knot色 + 20%背景色
- α = 0.2（20% 不透明）：最终色 = 20%knot色 + 80%背景色

### 3.5 局限性

Pass 1 的深度测试锁死了"只保留最近的面"。knot 的背面深度更大，Pass 2 被 `GL_LEQUAL` 深度测试挡住。所以：
- **正面能看到，背面看不到**。
- 简单物体够用，复杂物体（如 knot 的卷曲处）不够正确。

---

## 4. Depth Peeling（深度剥离 / OIT）

### 4.1 核心原理

"剥洋葱"思想：

```
第 1 次画 → 得到第 1 层（最近的），记下深度
第 2 次画 → 扔掉深度 ≤ 第 1 层的片元 → 得到第 2 层，记下深度
第 3 次画 → 扔掉深度 ≤ 第 2 层的片元 → 得到第 3 层
...
每剥出一层就混合到累积 buffer 里
没了新片元就停
```

### 4.2 数据结构

```
_colorBlendFbo       ← 累积所有层混合结果（用 depthTextures[0] 当深度附件）
_fbos[0], _fbos[1]  ← Ping-Pong FBO（每层轮换读写）
  ├─ _colorTextures[i]  ← 存这一层的颜色
  └─ _depthTextures[i]  ← 存这一层的深度
```

### 4.3 步 1：Init（初始化，画第一层）

`oit_init.frag`：

```glsl
void main() {
    vec3 premultiedColor = lambertShading() * material.transparent;  // 预乘 alpha
    color = vec4(premultiedColor, 1.0f - material.transparent);
    //                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                           alpha = 还能穿透多少光（20% = 还能看到后面）
}
```

**预乘 Alpha（Premultiplied Alpha）**是为什么？

普通混合（back-to-front）：`C = src × α + dst × (1-α)`
Front-to-back 需要：`C = dst + src × dst_α`，`α_new = α_old × (1-α_src)`

预乘颜色（`C × α`）让 front-to-back 的 RGB 可以直接加。

### 4.4 步 2：Peeling 循环

`oit_peel.frag`：

```glsl
void main() {
    // ★ 核心：深度 ≤ 上一层 → 丢弃
    if (gl_FragCoord.z <= getPeelingDepth()) {
        discard;
    }
    vec3 premultiedColor = lambertShading() * material.transparent;
    color = vec4(premultiedColor, material.transparent);
}
```

`getPeelingDepth()` 从上一层的深度纹理读取：

```glsl
float getPeelingDepth() {
    float u = gl_FragCoord.x / float(windowExtent.width);
    float v = gl_FragCoord.y / float(windowExtent.height);
    return texture(depthTexture, vec2(u, v)).r;
}
```

### 4.5 CPU 端循环

```cpp
int readBuffer = 0, writeBuffer = 1;
const int MAX_LAYERS = 16;

for (int layer = 1; layer < MAX_LAYERS; ++layer) {
    // 2.1 Peeling Pass — 剥出新一层
    glBeginQuery(GL_SAMPLES_PASSED, _queryId);    // ★ 数有多少片元通过
    _fbos[writeBuffer]->bind();
    _depthTextures[readBuffer]->bind(0);           // 用上一层深度做 discard
    _knot->draw();
    glEndQuery(GL_SAMPLES_PASSED);

    GLuint samplesPassed = 0;
    glGetQueryObjectuiv(_queryId, GL_QUERY_RESULT, &samplesPassed);
    if (samplesPassed == 0) break;                // ★ 没有新片元，停止

    // 2.2 Blending Pass — 把新层混合到累积 buffer
    _colorBlendFbo->bind();
    // Front-to-Back Under-Blending:
    glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE,
                        GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
    _fullscreenQuad->draw();

    std::swap(readBuffer, writeBuffer);           // Ping-Pong
}
```

**Occlusion Query** 的作用：`GL_SAMPLES_PASSED` 统计通过的片元数。如果某次剥离没有新片元了 → 0 → break，优雅停止。

### 4.6 混合方程的数学含义

Front-to-Back Under-Blending：

$$RGB_{new} = RGB_{old} \cdot \alpha_{old} + RGB_{layer}$$
$$\alpha_{new} = \alpha_{old} \cdot (1 - \alpha_{layer})$$

举例（每层 80% 不透明，即 α_layer = 0.8）：

```
第 1 层后：α = 0.2  (20% 光还能穿透)
第 2 层后：α = 0.04 (4%)
第 3 层后：α = 0.008 (0.8%)
第 4 层后：α ≈ 0
```

### 4.7 步 3：Final Pass（混合背景）

`oit_final.frag`：

```glsl
void main() {
    vec4 frontColor = texture(blendTexture, uv);  // 累积的透明层
    // 背景按剩余透明度"漏"过来
    color = vec4((frontColor + backgroundColor * frontColor.a).rgb, 1.0f);
}
```

### 4.8 Depth Peeling 的完整数据流

```
Init: _knot → _colorBlendFbo（第 1 层颜色 + 深度）
       ↓
Layer 1: _depthTextures[0] + _knot → _fbos[1]（剥第 2 层）
          _fbos[1].color → blend → _colorBlendFbo（累加）
       ↓
Layer 2: _depthTextures[1] + _knot → _fbos[0]（剥第 3 层）
          _fbos[0].color → blend → _colorBlendFbo（累加）
       ↓
... (occlusion query → 0，停止)
       ↓
Final: _colorBlendFbo + 背景色 → 屏幕
```

---

## 5. 动手实操

### 5.1 编译运行

```powershell
cmake --build build --target bonus1 --config Release
.\build\bin\Release\bonus1.exe
```

### 5.2 ImGui 操作

- **Alpha Testing**：观察 `transparent.png` 的透明区域被 discard，注意边缘锯齿
- **Alpha Blending**：拖动 `transparent` 滑条（0~1）观察渐变的半透明。试着调极端值感受差异
- **Depth Peeling**：对比 Alpha Blending，knot 的背面部也能看到

### 5.3 实验建议

1. 把 `alpha_test.frag` 的阈值从 0.5 改成 0.1 或 0.9 → 观察 discard 范围变化
2. 把 `GL_LEQUAL` 删掉 → 体验"什么都看不到"的经典 bug
3. 把 MAX_LAYERS 从 16 改成 3 → 观察剥离不完整的效果
4. 修改背景色 → 观察透明混合在不同背景上的表现

---

## 6. 知识延伸

- **Weighted Blended OIT**：工业界更常用的 OIT 近似方案，单 pass，用权重近似替代排序
- **Alpha to Coverage**：用 MSAA 的 coverage mask 模拟透明，比 Alpha Testing 平滑
- **Premultiplied Alpha 为什么重要**：普通 alpha 在插值和滤波时会产生"黑边"，预乘可以避免
