---
comments : true
---

# Bonus4 — 阴影映射（Shadow Mapping）

!!! tip "核心要点"
    阴影 = 从光源视角渲染一张深度图，回到主相机画场景时，检查每个片元在光源视角下是否被挡住。三种方案层层递进：方向光阴影 → 点光源 Cubemap 阴影 → 级联阴影贴图（CSM）。

## 0. 前置知识

- **Shadow Map（阴影贴图）**：从光源位置渲染场景，只存深度。主渲染时，把片元变换到光源空间，比较它的深度和 shadow map 的深度 → 更深 = 在阴影中。
- **PCF（Percentage Closer Filtering）**：采样周围多个 shadow map 纹素取平均，产生软阴影。
- **Cubemap**：6 面纹理组成的"天空盒"，用方向向量采样。用于点光源的全向阴影。
- **级联阴影（CSM）**：把视锥体分成多个区域（近、中、远），每个区域一张不同精度的 shadow map。

## 1. 整体架构

```
每帧流程：
1. 渲染 Shadow Maps
   ├─ 方向光深度图 → _depthFbo + _depthTexture
   ├─ 点光源 Cubemap → _depthCubeTexture（6 面）
   └─ CSM → _depthTextureArray（5 层，不同视锥区域）
2. 渲染场景（lambert.frag）
   ├─ 对每个片元：检查方向光阴影 + 点光源阴影 + CSM 阴影
   └─ ambient + diffuse * shadowFactor
3. Debug 模式：单独显示各深度图
```

## 2. 方向光阴影（Directional Shadow Map）

### 2.1 步骤 1：从光源渲染深度图

```cpp
// 计算光源空间矩阵（类似相机，但"看的方向"是光照方向）
glm::mat4 lightProjection = glm::ortho(-50, 50, -50, 50, 0.1, 200);
glm::mat4 lightView = glm::lookAt(lightPos, lightPos + lightDir, up);
glm::mat4 lightSpaceMatrix = lightProjection * lightView;

// 渲染到 _depthFbo 的深度纹理
_depthFbo->bind();
glClear(GL_DEPTH_BUFFER_BIT);
// 用 directional_depth shader 从光源视角画场景
renderSceneFromLight(*_directionalDepthShader);
```

### 2.2 深度着色器

```glsl
// directional_depth.vert — 输出片元在光源空间的位置
uniform mat4 lightSpaceMatrix;
uniform mat4 model;
void main() {
    gl_Position = lightSpaceMatrix * model * vec4(aPosition, 1.0);
}

// directional_depth.frag
void main() {
    // gl_FragDepth = gl_FragCoord.z;  // TODO: 需要取消注释
}
```

### 2.3 步骤 2：主渲染时检查阴影

```glsl
// lambert.frag
uniform mat4 directionalLightSpaceMatrix;
uniform sampler2D depthTexture;

float shadowFactor = 1.0;  // 1.0 = 全亮, 0.0 = 全暗

// 1. 把世界空间位置变换到光源空间
vec4 lightSpacePos = directionalLightSpaceMatrix * vec4(fPosition, 1.0);
vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;  // 透视除法
projCoords = projCoords * 0.5 + 0.5;  // [-1,1] → [0,1]

// 2. 比较深度
float closestDepth = texture(depthTexture, projCoords.xy).r;
float currentDepth = projCoords.z;
if (currentDepth > closestDepth + bias) {
    shadowFactor = 0.0;  // 在阴影中
}

// 3. 光照 = 环境光 + 漫反射 * shadowFactor
```

### 2.4 Shadow Acne 和 Bias

**Shadow Acne（阴影痤疮）**：由于 shadow map 精度有限，被照亮的面自己也会产生自阴影条纹。

解决方案：加一个 **bias**（偏移量）：
```glsl
float bias = 0.005;
if (currentDepth - bias > closestDepth) { shadowFactor = 0.0; }
```

### 2.5 PCF 软阴影

```glsl
uniform int directionalFilterRadius;  // 0~5

float shadow = 0.0;
vec2 texelSize = 1.0 / textureSize(depthTexture, 0);
for (int x = -radius; x <= radius; x++) {
    for (int y = -radius; y <= radius; y++) {
        float depth = texture(depthTexture, projCoords.xy + vec2(x,y) * texelSize).r;
        shadow += (currentDepth - bias > depth) ? 1.0 : 0.0;
    }
}
shadow /= ((2*radius+1) * (2*radius+1));
```

半径越大 → 阴影边缘越模糊 → 越像真实世界的软阴影。

---

## 3. 点光源全向阴影（Omnidirectional Shadow）

### 3.1 为什么需要 Cubemap

点光源向**所有方向**发光。单张 2D 纹理只能覆盖一个方向。

解决方案：渲染 6 次（+X, -X, +Y, -Y, +Z, -Z），存到 Cubemap 的 6 个面。

### 3.2 渲染 Cubemap 深度

```cpp
// 6 个面的投影矩阵（90° FOV，覆盖全向）
glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, zFar);

// 6 个 view 矩阵（看向不同方向）
glm::mat4 views[6] = {
    lookAt(lightPos, lightPos + (+1,0,0), (0,-1,0)),  // +X
    lookAt(lightPos, lightPos + (-1,0,0), (0,-1,0)),  // -X
    lookAt(lightPos, lightPos + (0,+1,0), (0,0,+1)),  // +Y
    lookAt(lightPos, lightPos + (0,-1,0), (0,0,-1)),  // -Y
    lookAt(lightPos, lightPos + (0,0,+1), (0,-1,0)),  // +Z
    lookAt(lightPos, lightPos + (0,0,-1), (0,-1,0)),  // -Z
};

// 渲染 6 次，每次绑 Cubemap 的一个面
for (int i = 0; i < 6; i++) {
    _depthCubeFbos[i]->bind();  // 每个面一个 FBO
    _pointLightSpaceMatrices[i] = shadowProj * views[i];
    renderSceneFromLight(*_omnidirectionalDepthShader);
}
```

### 3.3 深度计算

点光源的"深度"不是坐标的 z 值，而是**到光源的欧氏距离**：

```glsl
// omnidirectional_depth.frag
uniform vec3 lightPosition;
uniform float zFar;

void main() {
    float distance = length(fPosition.xyz - lightPosition);
    gl_FragDepth = distance / zFar;  // 归一化到 [0, 1]
}
```

### 3.4 主渲染时采样

```glsl
uniform samplerCube depthCubeTexture;
uniform float pointLightZfar;

// 点光源方向（片元 → 光源）
vec3 fragToLight = pointLight.position - fPosition;
float currentDepth = length(fragToLight) / pointLightZfar;

// Cubemap 采样用方向向量（不需要 UV）
float closestDepth = texture(depthCubeTexture, -fragToLight).r;

float shadow = (currentDepth > closestDepth + bias) ? 0.0 : 1.0;
```

---

## 4. 级联阴影贴图（Cascade Shadow Maps）

### 4.1 问题

单张 shadow map 覆盖整个场景。近处阴影精度不够（一个纹素覆盖范围太大），远处阴影精度浪费。

### 4.2 方案

把视锥体沿 z 轴切成 5 段，每段单独一张 shadow map：

```
级联 0: 0~near   → 小范围，高精度  (512×512)
级联 1: near~mid1 → 中等范围
级联 2: mid1~mid2
级联 3: mid2~far1
级联 4: far1~far  → 大范围，低精度
```

### 4.3 级联分割

```cpp
// 计算相机视锥体在 z 轴上的分割点
std::vector<float> getCascadeDistances() const {
    float near = _camera->getNear();
    float far = _camera->getFar();
    // 对数分割（近处更密，远处更疏）
    for (int i = 0; i < cascadeLevels; i++) {
        float p = (i + 1) / float(cascadeLevels);
        float logSplit = near * pow(far / near, p);
        float uniSplit = near + (far - near) * p;
        distances[i] = lerp(uniSplit, logSplit, 0.5);  // 混合均匀和对数
    }
}
```

### 4.4 渲染 CSM

```cpp
// 每层级联：
// 1. 计算该级相机的子视锥体
// 2. 子视锥体的包围盒 → 光源空间的 ortho 投影矩阵
// 3. 渲染到 texture array 的对层级

for (int i = 0; i < cascadeLevels; i++) {
    _depthCascadeFbos[i]->bind();
    // _directionalLightSpaceMatrices[i] = lightProj * lightView（不同范围）
    renderSceneFromLight(*_directionalDepthShader);
}
```

### 4.5 主渲染时选择级联

```glsl
// lambert.frag
uniform sampler2DArray depthTextureArray;
uniform mat4 directionalLightSpaceMatrices[16];
uniform float cascadeZfars[16];
uniform int cascadeCount;

// 1. 确定片元在哪个级联
vec4 viewSpacePos = view * vec4(fPosition, 1.0);
float viewDepth = -viewSpacePos.z;
int cascadeIdx = 0;
for (int i = 0; i < cascadeCount - 1; i++) {
    if (viewDepth > cascadeZfars[i]) cascadeIdx = i + 1;
}

// 2. 用该级联的 light space matrix + 对应层的 depth texture
vec4 lightSpacePos = directionalLightSpaceMatrices[cascadeIdx] * vec4(fPosition, 1.0);
// ...
float closestDepth = texture(depthTextureArray, vec3(projCoords.xy, cascadeIdx)).r;
```

---

## 5. 调试功能

Debug 视图可以单独显示每个 shadow map / cascade：

```cpp
enum class DebugView {
    None,
    DirectionalLightDepthTexture,     // 方向光深度图
    PointLightDepthTexture,           // 点光源 Cubemap
    CascadeDepthTextureLevel0,        // CSM 第 0 级
    CascadeDepthTextureLevel1,        // CSM 第 1 级
    // ...
};
```

用全屏四边形（`quad.frag` / `quad_csm.frag`）把深度纹理读出来显示。

---

## 6. 动手实操

### 6.1 编译运行

```powershell
cmake --build build --target bonus4 --config Release
.\build\bin\Release\bonus4.exe
```

### 6.2 实验

1. **基本阴影**：兔子在地面上的方向光阴影
2. **PCF 半径**：ImGui 调 `directionalFilterRadius`（0~5），观察硬阴影→软阴影的过渡
3. **点光源阴影**：开启 `enableOmnidirectionalPCF`
4. **CSM**：开启 `enableCascadeShadowMapping`，调整 5 级 cascade bias，观察近处高精度阴影
5. **Debug 视图**：切换 DebugView 看各个深度图和 cascade 级别

### 6.3 核心 TODO

`lambert.frag` 的 `main()` 函数目前直接输出无阴影的光照：

```glsl
void main() {
    vec3 ambient = calcAmbientLight();
    vec3 diffuse = calcDirectionalLight(normal) + calcPointLight(normal);
    color = vec4(ambient + diffuse, 1.0);  // TODO: 乘 shadowFactor
}
```

需要加上：
- 方向光 shadow check（sampler2D depthTexture）
- 点光源 shadow check（samplerCube depthCubeTexture）
- CSM 选择 + sampler2DArray 采样

---

## 7. 知识延伸

- **Variance Shadow Maps (VSM)**：存深度和深度²，可以用硬件滤波，比 PCF 更快
- **Exponential Shadow Maps (ESM)**：存指数深度，同样可硬件滤波
- **Contact Hardening Shadows**：近处阴影硬、远处阴影软（符合物理直觉的 PCF 变体）
- **Screen-Space Shadows**：在后处理阶段用深度缓冲近似计算阴影（低开销，但精度有限）
