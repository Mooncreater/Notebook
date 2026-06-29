---
comments : true
---

# Bonus3 — 后处理特效（SSAO + Bloom）

!!! tip "核心要点"
    后处理 = 先正常渲染场景到 G-Buffer（位置、法线、颜色），再用全屏四边形做逐像素特效。SSAO 给暗角加阴影立体感，Bloom 让亮的地方发光溢色。

## 0. 前置知识

- **延迟渲染（Deferred Rendering）**：第一步（Geometry Pass）不输出最终颜色，而是把位置、法线、颜色分别存到多个纹理（G-Buffer）。第二步（Lighting Pass）用 G-Buffer 计算光照。好处：光照计算只对可见像素做，复杂场景高效。
- **全屏四边形（Fullscreen Quad）**：一个铺满屏幕的矩形，把纹理贴上去。后处理特效的标准载体。
- **Framebuffer（FBO）**：OpenGL 可以渲染到内存纹理而非屏幕。本 bonus 用了多个 FBO。

## 1. 整体架构

```
Geometry Pass → G-Buffer [位置 | 法线 | 颜色 | 深度]
    ↓
┌── SSAO Pass ──────────────┐   ┌── Bloom Pass ───────────────┐
│ G-Buffer 法线+深度         │   │ 提取亮色 (extractBrightColor) │
│   → 计算遮蔽因子 (ssao)    │   │   → 高斯模糊 (gaussianBlur)   │
│   → 模糊 (ssaoBlur)       │   │   → 叠加回原图               │
└───────────────────────────┘   └──────────────────────────────┘
    ↓                              ↓
Lighting Pass (ssao_lighting)   叠加
    ↓
    最终帧 → 屏幕
```

## 2. Geometry Pass

### 2.1 G-Buffer 结构

四个纹理，每个和屏幕一样大：

```
_gPosition : GL_RGBA32F  → 视图空间的位置 (x,y,z)
_gNormal   : GL_RGBA32F  → 视图空间的法线 (nx,ny,nz)
_gAlbedo   : GL_RGBA32F  → 物体表面颜色 (r,g,b)
_gDepth    : GL_DEPTH    → 深度
```

### 2.2 着色器

```glsl
// geometry.vert — 变换到视图空间
void main() {
    vec4 viewSpacePos = view * model * vec4(aPosition, 1.0f);
    position = viewSpacePos.xyz;
    normal = normalize(mat3(transpose(inverse(view * model))) * aNormal);
    gl_Position = projection * viewSpacePos;
}

// geometry.frag — 输出到多个纹理
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec3 gAlbedo;

void main() {
    gPosition = position;
    gNormal = gl_FrontFacing ? normal : -normal;  // 背面法线翻转
    gAlbedo = vec3(0.95f);  // 统一米白色
}
```

!!! tip "为什么存视图空间而不是世界空间？"
    后续光照计算（Lambert、SSAO 采样）在视图空间更方便——相机就是原点，方向和距离计算简化。

---

## 3. SSAO（屏幕空间环境光遮蔽）

### 3.1 核心思想

看每个像素周围的几何体"围"得有多紧：
- 被围得紧 → 光线不容易进来 → 更暗
- 空旷开阔 → 光线充足 → 更亮

### 3.2 算法步骤

```
对每个像素 p，在 G-Buffer 中获取它的位置 P 和法线 N：
  1. 在以 P 为中心、法线朝向的半球内随机采样 64 个点
  2. 对每个采样点 S：
     a. 从 G-Buffer 深度重建 S 的世界/视图位置
     b. 如果 S 的实际深度 < S 的采样位置（说明有东西挡着）→ occlusion += 1
  3. occlusion / 64 → 遮蔽因子
  4. 1 - 遮蔽因子 → SSAO 值
```

### 3.3 着色器（TODO: ssao.frag）

```glsl
const int nSamples = 64;
const float radius = 1.0f;

uniform sampler2D gPosition;   // G-Buffer 位置
uniform sampler2D gNormal;     // G-Buffer 法线
uniform sampler2D noiseMap;    // 4×4 随机旋转纹理（消除采样规律性）
uniform vec3 sampleVecs[64];   // 64 个随机半球方向
uniform mat4 projection;

void main() {
    // TODO: 实现 SSAO
    
    // 思路：
    // 1. 从 gPosition 和 gNormal 读当前像素的位置和法线
    // 2. 用 noiseMap 生成切线空间的随机旋转
    // 3. 对 64 个采样方向：
    //    a. 把方向从切线空间转到视图空间
    //    b. offset = P + direction * radius
    //    c. 把 offset 投影到屏幕空间，从 gPosition 读取实际深度
    //    d. 比较：实际深度 vs offset 深度
    // 4. occlusion = 未遮蔽数 / 64
    // 5. ssaoResult = occlusion
    
    ssaoResult = 1.0f;  // 默认无遮蔽
}
```

### 3.4 SSAO 模糊（ssao_blur.frag）

原始 SSAO 有噪点，做一个 5×5 的 box blur 平滑：

```glsl
void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(ssaoResult, 0));
    float result = 0.0;
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result += texture(ssaoResult, screenTexCoord + offset).r;
        }
    }
    blurResult = result / 25.0;  // 5×5 平均
}
```

### 3.5 SSAO Lighting（TODO: ssao_lighting.frag）

结合 SSAO 遮蔽因子做最终光照：

```glsl
void main() {
    // 从 G-Buffer 读数据
    vec3 position = texture(gPosition, screenTexCoord).xyz;
    vec3 normal = texture(gNormal, screenTexCoord).xyz;
    vec3 albedo = texture(gAlbedo, screenTexCoord).rgb;
    float occlusion = texture(ssaoResult, screenTexCoord).x;
    
    // TODO: Lambert 漫反射 + 衰减点光源
    // 光照公式：color = ambient + diffuse * occlusion
    // ambient 很小（模拟被遮蔽的环境光）
    // diffuse = max(dot(N, L), 0) * lightColor * intensity / distance²
    
    fragColor = vec4(normal, 1.0f);  // 当前调试：显示法线
}
```

---

## 4. Bloom（泛光）

### 4.1 核心思想

现实世界中，非常亮的光源（太阳、灯泡）会让周围的像素也变亮，产生"光晕"效果。

Bloom 的三步法：
```
1. 提取亮部（亮度 > 阈值的像素）
2. 高斯模糊（让亮部向周围扩散）
3. 叠加回原图
```

### 4.2 提取亮色（TODO: extract_bright_color.frag）

```glsl
void main() {
    // TODO: 提取亮度超过 1.0 的颜色分量
    // vec3 color = texture(sceneMap, screenTexCoord).rgb;
    // float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722)); // 感知亮度
    // if (brightness > 1.0) {
    //     brightColorMap = vec4(color, 1.0);
    // } else {
    //     brightColorMap = vec4(0.0);
    // }
    
    brightColorMap = vec4(0.0f);
}
```

### 4.3 高斯模糊（TODO: gaussian_blur.frag）

两 pass 可分离高斯模糊（先水平，再垂直），每个像素 = 周围 5 个像素的加权平均：

```glsl
const float weight[5] = float[](
    0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);

uniform bool horizontal;  // true = 水平模糊, false = 垂直模糊

void main() {
    // TODO: 实现可分离高斯模糊
    
    // 思路：
    // vec2 texelSize = 1.0 / textureSize(image, 0);
    // vec3 result = texture(image, screenTexCoord).rgb * weight[0];
    // for (int i = 1; i < 5; i++) {
    //     vec2 offset = horizontal ? vec2(texelSize.x * i, 0.0) : vec2(0.0, texelSize.y * i);
    //     result += texture(image, screenTexCoord + offset).rgb * weight[i];
    //     result += texture(image, screenTexCoord - offset).rgb * weight[i];
    // }
    // FragColor = vec4(result, 1.0);
    
    FragColor = texture(image, screenTexCoord);
}
```

### 4.4 叠加（blend_bloom_map.frag）

```glsl
void main() {
    vec3 sceneColor = texture(scene, screenTexCoord).rgb;
    vec3 bloomColor = texture(bloomBlur, screenTexCoord).rgb;
    FragColor = vec4(sceneColor + bloomColor, 1.0);  // 直接相加
}
```

---

## 5. 动手实操

### 5.1 编译运行

```powershell
cmake --build build --target bonus3 --config Release
.\build\bin\Release\bonus3.exe
```

### 5.2 实验

1. **开/关 SSAO**：ImGui 切换，观察兔子耳朵和身体交界处、方块边角的明暗变化
2. **开/关 Bloom**：观察点光源周围的泛光光晕
3. **逐一完成 TODO**：
   - `ssao.frag`：最核心的 SSAO 采样逻辑
   - `ssao_lighting.frag`：把遮蔽因子代入光照
   - `extract_bright_color.frag`：亮度阈值提取
   - `gaussian_blur.frag`：可分离高斯模糊

### 5.3 调试技巧

- 单独显示 G-Buffer 的各个通道：位置 = RGB 查看，法线 = 方向转颜色，深度 = 灰度
- 单独显示 SSAO 遮蔽纹理 → 暗角应该出现在几何体交接处
- 单独显示 Bloom 模糊纹理 → 应该是亮部扩散的"光团"

---

## 6. 知识延伸

- **HBAO / HDAO**：比 SSAO 更精确的环境光遮蔽算法（考虑地平线角）
- **SSGI**：屏幕空间全局光照，让颜色在像素间"弹跳"
- **Tone Mapping**：Bloom 通常配合 HDR + Tone Mapping 使用（先存高动态范围，最后映射到 LDR）
- **Depth of Field**：另一个经典后处理——远处模糊，近处清晰
