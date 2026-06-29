---
comments : true
---

# Bonus5 — GPU 光线追踪（Ray Tracing）

!!! tip "核心要点"
    在 GPU 上用**全屏四边形片元着色器**实现路径追踪。所有场景数据（几何体、BVH、材质、顶点）打包进 2D 纹理传给 shader。逐像素发射光线，递进累积采样，支持 Lambertian / Metal / Dielectric 三种材质。

## 0. 前置知识

- **光线追踪 vs 光栅化**：光栅化是"把三角形投影到屏幕"；光线追踪是"从眼睛发射光线，打到什么就算什么"。光线追踪天然支持反射、折射、软阴影。
- **路径追踪（Path Tracing）**：光线打到物体后继续弹射，每次弹射随机采样方向，大量采样取平均 → 收敛到真实光照。
- **BVH（包围体层次结构）**：树状结构把三角形按空间分组，求交从 O(n) 降到 O(log n)。
- **蒙特卡洛积分**：用随机采样近似积分。路径追踪本质是在解渲染方程（Kajiya 方程）的蒙特卡洛估计。

## 1. 为什么不用 Compute Shader

本 bonus 使用最"朴素"的 GPU 计算方式：**全屏四边形片元着色器**。

```
屏幕 = 1200 × N 像素
每个像素 → 一个 fragment shader 调用 → 各自独立追踪一条光线
```

限制：GL 3.3 没有 SSBO（Shader Storage Buffer Object），所以所有数据（场景几何体、材质、BVH）必须打包进**2D 纹理**。

---

## 2. 数据打包：纹理即数组

### 2.1 核心技巧

GPU 本来的设计是用纹理存颜色。这里把纹理当通用数组用：

```
固定宽度 = 2048 texels
高度 = ceil(nObjects * objTexels / 2048)
```

| 纹理 | 格式 | 每个对象占 | 内容 |
|------|------|-----------|------|
| `_sphereBuffer` | RGBA32F | 1 texel | (pos.xyz, radius) |
| `_vertexBuffer` | RGBA32F | 2 texels/顶点 | (pos, norm.x) + (norm.yz, texCoord) |
| `_indexBuffer` | RGB32I | 1 texel/三角形 | (v0, v1, v2) |
| `_materialBuffer` | RGBA32F | 2 texels/材质 | (type, ior, fuzz) + albedo |
| `_primitiveBuffer` | RGB32I | 1 texel/图元 | (shapeType, shapeIdx, materialIdx) |
| `_bvhBuffer` | RGBA32F | 3 texels/节点 | pMin + pMax + (type, child1, child2) |

### 2.2 1D → 2D 索引转换

```glsl
vec2 getSampleIdx(sampler2D data, int idx) {
    ivec2 texSize = textureSize(data, 0);
    int x = idx % texSize.x;         // 宽度方向
    int y = idx / texSize.x;         // 高度方向
    return vec2((x + 0.5) / texSize.x, (y + 0.5) / texSize.y);  // +0.5 = 像素中心
}
```

### 2.3 Int → Float 的位转换技巧

因为 `GL_RGBA32F` 只能存浮点数，int 值通过 `intBitsToFloat` 存进去，读出来用 `floatBitsToInt` 还原：

```cpp
// C++ 端
static int toFloatLayout(int v) {
    float f;
    memcpy(&f, &v, sizeof(float));  // 逐位复制
    return *reinterpret_cast<int*>(&f);  // 返回 float 位模式当 int
}
```

---

## 3. 渐进式渲染（Progressive Rendering）

### 3.1 Ping-Pong Framebuffer

```
Frame N:
  读 _outFrames[0]（前 N-1 帧的累积） + _rngStates[0]（旧 RNG 状态）
  画全屏四边形（1 sample/pixel）
  写 _outFrames[1]（更新累积）    + _rngStates[1]（新 RNG 状态）
  swap(0, 1)

Frame N+1:
  读 _outFrames[1]（前 N 帧累积）
  写 _outFrames[0]
  swap(0, 1)
```

### 3.2 运行平均

```glsl
void outputSample(vec4 color) {
    vec3 rst = texture(RTResult, screenTexCoord).rgb;  // 旧累积
    vec3 linear = (inverseGammaCorrection(rst) * totalSamples + color.rgb) 
                  / (totalSamples + 1);                 // 平均
    fragColor = vec4(gammaCorrection(linear), 1.0);
}
```

第 1 帧：全是噪点。第 100 帧：逐渐收敛。第 1000 帧：干净。

### 3.3 每像素独立 RNG

```glsl
// RNG 状态存在 R32UI 纹理里，每帧读-用-写
void rngInit() {
    rngState = texture(oldRngState, screenTexCoord).r;
}

float rngGetRandom1D() {
    rngState ^= (rngState << 13);    // XOR-shift 算法
    rngState ^= (rngState >> 17);
    rngState ^= (rngState << 5);
    return min(0.99999994, float(rngState) * (1.0 / 4294967296.0));
}

void main() {
    rngInit();
    // ... 追踪使用 rngGetRandom1D / rngGetRandom2D ...
    fragRngState = rngState;  // 写回
}
```

---

## 4. 相机与光线生成

### 4.1 相机模型

```glsl
struct Camera {
    mat4 cameraToWorld;    // view 矩阵的逆（相机位置 → 世界空间）
    mat4 rasterToCamera;   // 屏幕像素 → 相机空间
};
```

### 4.2 生成光线（带抖动抗锯齿）

```glsl
Ray generateRay(vec2 u) {
    // u = 随机偏移 [0,1)² → 抖动抗锯齿
    vec2 pixelPos = gl_FragCoord.xy + u;
    
    // 屏幕 → 相机空间 → 世界空间
    vec4 localDir = rasterToCamera * vec4(pixelPos, 0.0, 1.0);
    vec4 worldDir = cameraToWorld * vec4(normalize(localDir.xyz), 0.0);
    
    Ray ray;
    ray.o = (cameraToWorld * vec4(0, 0, 0, 1)).xyz;  // 世界空间相机位置
    ray.dir = normalize(worldDir.xyz);
    ray.tMax = INFINITY;
    return ray;
}
```

---

## 5. BVH 加速结构

### 5.1 为什么需要 BVH

场景可能有几千个三角形 + 几百个球体。逐图元求交 → 每像素每光线检查几千次 → 太慢。

BVH（Bounding Volume Hierarchy）用树状结构把图元按空间分组：

```
         [Root AABB: 整个场景]
         /                    \
  [Left: 左半边]        [Right: 右半边]
    /        \            /          \
 [Leaf]    [Leaf]    [Leaf]      [Leaf]
 (图元1-3) (图元4-6)  (图元7-9)    (图元10-12)
```

光线求交时：先测父节点 AABB，不交 → 整棵子树跳过；交 → 递归下去。

### 5.2 BVH 节点结构

```glsl
struct BVHNode {
    AABB box;
    int nodeType;   // 0=内部节点, 1=叶节点
    int firstVal;   // 内部节点=左子, 叶节点=起始图元索引
    int secondVal;  // 内部节点=右子, 叶节点=图元数量
};
```

### 5.3 GPU 端的 BVH 遍历（TODO）

```glsl
bool intersect(inout Ray ray, inout Interaction isect) {
    // TODO: 实现栈式 BVH 遍历
    
    // 伪代码：
    // int stack[64]; int sp = 0;
    // stack[sp++] = 0;  // 从根节点开始
    // while (sp > 0) {
    //     int nodeIdx = stack[--sp];
    //     BVHNode node = getBVHNode(bvh, nodeIdx);
    //     if (!intersectAABB(ray, node.box)) continue;
    //     if (node.nodeType == LEAF) {
    //         对 node 中的每个图元做 intersectPrimitive
    //     } else {
    //         stack[sp++] = node.secondVal;  // 右子
    //         stack[sp++] = node.firstVal;   // 左子
    //     }
    // }
}
```

### 5.4 AABB 求交（Slab 方法）

```glsl
bool intersectAABB(inout Ray ray, inout AABB box, inout vec3 invDir) {
    // 对 x, y, z 三个轴分别计算进入和离开的 t
    float tMin = (box.pMin.x - ray.o.x) * invDir.x;
    float tMax = (box.pMax.x - ray.o.x) * invDir.x;
    // ... 处理方向为负时交换 tMin/tMax ...
    // 三个轴的交集 → [tEnter, tExit]
    // tEnter < tExit && tExit > 0 → 相交
}
```

---

## 6. 路径追踪循环（TODO: trace()）

```glsl
vec4 trace(inout Ray ray) {
    vec3 attenuation = vec3(1.0);  // 累积衰减
    vec3 radiance = vec3(0.0);     // 累积光能
    
    for (int depth = 0; depth < maxTraceDepth; depth++) {
        Interaction isect;
        if (!intersect(ray, isect)) {
            radiance += attenuation * texture(sky, ray.dir).rgb;
            break;  // 打到天空 → 停止
        }
        
        // 俄罗斯轮盘赌（可选优化：概率终止递归）
        
        switch (isect.material.type) {
        case LAMBERTIAN_MATERIAL:
            if (!lambertianScatterFunction(ray, isect)) break;
            break;
        case METAL_MATERIAL:
            if (!metalScatterFunction(ray, isect)) break;
            break;
        case DIELECTRIC_MATERIAL:
            dielectricScatterFunction(ray, isect);
            break;
        }
        
        attenuation *= isect.material.albedo;  // 颜色累积衰减
    }
    return vec4(radiance, 1.0);
}
```

---

## 7. 材质散射函数（全部 TODO）

### 7.1 Lambertian（漫反射）

```glsl
bool lambertianScatterFunction(inout Ray ray, inout Interaction isect) {
    // TODO
    // 1. 在 hitPoint 处建立局部坐标系（法线 = z）
    // 2. 用 cosine-weighted 半球采样生成随机方向
    //    方向在局部空间中，再转到世界空间
    // 3. 新光线：原点 = hitPoint + epsilon * normal（防自交）
    //           方向 = 采样的散射方向
    // 4. return true
}
```

**为什么用 cosine-weighted 采样？**
Lambert 材质向各个方向反射的概率正比于 cos(θ)，cosine-weighted 采样 = 重要性采样，收敛更快。

### 7.2 Metal（金属反射）

```glsl
bool metalScatterFunction(inout Ray ray, inout Interaction isect) {
    // TODO
    // 1. 计算反射方向：reflected = reflect(ray.dir, normal)
    // 2. 加模糊：reflected + fuzz * randomInUnitSphere()
    // 3. 新光线方向 = reflected（如果 fuzz > 0 则加扰动）
    // 4. 如果新方向指向面内（dot < 0），return false（吸收）
    // 5. return true
}
```

### 7.3 Dielectric（电介质 / 玻璃）

```glsl
void dielectricScatterFunction(inout Ray ray, inout Interaction isect) {
    // TODO
    // 1. 计算折射率比：eta = 入射介质ior / 出射介质ior
    //    进入物体：eta = 空气(1.0) / 材质ior
    //    离开物体：eta = 材质ior / 空气(1.0)
    // 2. Schlick Fresnel 计算反射概率
    // 3. 随机数 < 反射概率 → 反射
    //    随机数 ≥ 反射概率 → 折射（Snell 定律，考虑全内反射）
    // 4. 设置新光线方向和原点
}
```

**Fresnel 效应**：入射角越大，反射越多。这也是为什么看水面远处比近处反光更强。

```glsl
float fresnelSchlick(float cosTheta, float ior) {
    float r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
}
```

---

## 8. 图元求交

### 8.1 球体

$$a t^2 + 2b t + c = 0$$

其中 $a = \mathbf{d} \cdot \mathbf{d}$，$b = (\mathbf{o} - \mathbf{c}) \cdot \mathbf{d}$，$c = (\mathbf{o} - \mathbf{c}) \cdot (\mathbf{o} - \mathbf{c}) - r^2$

### 8.2 三角形（Möller–Trumbore）

用重心坐标直接求解交点，不需要先算平面再测内外：

1. 边向量 $e_1 = v_1 - v_0$, $e_2 = v_2 - v_0$
2. $s = o - v_0$
3. $s_1 = d \times e_2$, $s_2 = s \times e_1$
4. $t = (s_2 \cdot e_2) / (s_1 \cdot e_1)$
5. $u = (s_1 \cdot s) / (s_1 \cdot e_1)$, $v = (s_2 \cdot d) / (s_1 \cdot e_1)$
6. 如果 $u \ge 0$, $v \ge 0$, $u+v \le 1$, $t > \epsilon$ → 相交

---

## 9. 动手实操

### 9.1 编译运行

```powershell
cmake --build build --target bonus5 --config Release
.\build\bin\Release\bonus5.exe
```

### 9.2 实验

程序有三个预置场景（ImGui 切换）：
- **Scene 1**：3 个球体（玻璃 + 金属 + 漫反射）→ 最简单，先完成这个
- **Scene 2**：~500 个球体（地面 + 随机 + 3 个大球）→ 测试 BVH 性能
- **Scene 3**：~500 球体 + 3 个 Lucy 模型（三角面片）→ 测试三角形求交 + BVH

### 9.3 开发顺序建议

```
1. ray generation          → 能看到天空盒
2. sphere intersection     → 能看到球体轮廓
3. lambert material        → 球体有颜色
4. triangle intersection   → 模型可见
5. metal material          → 镜面反射
6. dielectric material     → 玻璃折射
7. BVH traversal           → 场景 2/3 不卡
8. progressive rendering   → 噪点逐渐消失
```

### 9.4 调试技巧

- 把 `trace()` 先改成直接返回 `ray.dir` 的归一化颜色 → 确认光线方向正确
- 把 `intersectSphere` 的 hit 处显示法线 → 确认法线计算正确
- 先不加 `maxTraceDepth` 循环，只做 1 次弹射 → 确认基本求交和光照
- BVH 先不开启，暴力遍历验证求交 → 再开启 BVH 对比速度

---

## 10. 知识延伸

- **MIS（多重重要性采样）**：结合光源采样和 BRDF 采样，大幅降低噪点
- **NEE（Next Event Estimation）**：每次弹射都直接采样光源，而非等光线"碰巧"打到光源
- **Disney BSDF / Principled BSDF**：工业界的统一材质模型，一个参数集合覆盖所有材质
- **OptiX / DXR / Vulkan RT**：硬件光线追踪 API，用专用 RT core 加速
