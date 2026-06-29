---
comments : true
---

# Bonus2 — 视锥体裁剪（Frustum Culling）

!!! tip "核心要点"
    场景里 10000 个小行星，相机只能看到其中一小部分。视锥体裁剪 = 在画之前就丢掉视野外的物体，CPU 端用数学判断，GPU 端用 Transform Feedback 并行判断 + 间接绘制。

## 0. 前置知识

- **视锥体（Frustum）**：相机能看到的空间范围，由近平面、远平面和四个侧面围成的六面体。
- **AABB（轴对齐包围盒）**：包裹物体的最小长方体，六个面平行于坐标轴，用 (min, max) 两个点表示。
- **平面方程**：$Ax + By + Cz + D = 0$，点 (x,y,z) 代入得距离。负值 = 在平面背面。
- **Instanced Rendering**：一个 draw call 画 N 个相同模型，每个实例有自己的变换矩阵。

## 1. 为什么需要视锥体裁剪

### 1.1 问题场景

一个中心行星 + 10000 个随机分布的小行星。相机在场景中自由移动旋转，一次只能看到几十到几百个小行星。

如果画 10000 个：
- 每个 500 面 → 500 万三角形
- 大部分在屏幕外，GPU 顶点处理完后 → 裁剪阶段丢弃
- 浪费了大量顶点着色器时间

### 1.2 裁剪的本质

```
for each 物体:
    if 物体的包围盒和视锥体有交集:
        画它
    else:
        跳过
```

---

## 2. CPU 端裁剪

### 2.1 预计算包围盒

小行星模型（rock.obj）加载时提取 AABB：

```cpp
// 从模型顶点计算 AABB
BoundingBox box;  // { min: (x1,y1,z1), max: (x2,y2,z2) }
```

### 2.2 视锥体提取

每帧从相机的 view-projection 矩阵提取 6 个平面：

```cpp
// 视锥体 = 6 个平面的交集
// 左平面、右平面、底平面、顶平面、近平面、远平面
struct Frustum {
    Plane planes[6];  // 每个平面: normal + signedDistance
};
```

### 2.3 AABB vs 视锥体测试

对每个小行星：

```cpp
for (int i = 0; i < _amount; i++) {
    // 把 AABB 转换到世界空间
    BoundingBox worldBox = transform(localBox, _modelMatrices[i]);
    
    // 测试世界空间 AABB 和视锥体
    if (intersects(worldBox, frustum)) {
        _visibles.push_back(i);  // 在视野内，记录
    }
}
```

**AABB-平面测试**：取 AABB 的 8 个顶点，如果所有顶点都在某个平面背面 → 完全在外面 → 丢弃。

### 2.4 绘制

```cpp
// 只画可见的
for (int idx : _visibles) {
    _lambertShader->setUniformMat4("model", _modelMatrices[idx]);
    _asternoid->draw();  // 一个一个画
}
```

`_drawAsternoidCount` 显示实际画了多少个。

---

## 3. GPU 端裁剪

### 3.1 为什么要上 GPU

CPU 裁剪 10000 个物体是 O(n) 串行判断，虽然快但：
- 10000 次包围盒变换（矩阵乘法）
- 10000 × 6 次平面-AABB 测试
- 结果还要从 CPU 传回 GPU 去画

GPU 裁剪：所有物体的包围盒在顶点着色器里**并行**判断。

### 3.2 Transform Feedback

Transform Feedback = GPU 能把顶点着色器的输出捕获到 buffer 里，而不是继续走光栅化。

```cpp
// 1. 创建 Transform Feedback buffer
glGenBuffers(1, &_transformFeedbackResultBuffer);
glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, _transformFeedbackResultBuffer);

// 2. 指定捕获变量
const char* varyings[] = {"visible"};
glTransformFeedbackVaryings(program, 1, varyings, GL_INTERLEAVED_ATTRIBS);
```

### 3.3 frustum_culling.vert（TODO）

Vertex shader 对每个实例判断其包围盒是否在视锥体内：

```glsl
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in mat4 aInstanceMatrix;
flat out int visible;  // ★ Transform Feedback 捕获这个

struct BoundingBox { vec3 min; vec3 max; };
struct Plane { vec3 normal; float signedDistance; };
struct Frustum { Plane planes[6]; };

uniform BoundingBox boundingBox;    // 小行星的局部空间 AABB
uniform Frustum frustum;           // 当前帧的视锥体

void main() {
    // TODO: 实现 GPU 视锥体裁剪判断
    
    // 思路：
    // 1. 用 aInstanceMatrix 把 boundingBox 变换到世界空间
    // 2. 对 frustum 的 6 个平面：检查变换后的 AABB
    // 3. 如果 AABB 完全在任何一个平面背面 → visible = 0
    // 4. 否则 visible = 1
    visible = 1;
}
```

### 3.4 Indirect Draw（间接绘制）

Transform Feedback 结果存了每个物体的 visible 标记，但怎么只画 visible=1 的？

**Indirect Draw**：绘制参数（顶点数、实例数等）存在 GPU buffer 里，GPU 自己读取，CPU 不参与。

```cpp
struct DrawElementsIndirectCommand {
    unsigned int count;          // 顶点数
    unsigned int instanceCount;  // 实例数
    unsigned int firstIndex;
    int baseVertex;
    unsigned int baseInstance;
};
```

每个 visible=1 的物体生成一个 command，visible=0 的 instanceCount=0（不画）。

```cpp
glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 
    (void*)(_indirectDrawCmds.data()));
```

---

## 4. 显示包围盒

### 4.1 调试功能

绿色线框显示每个小行星的 AABB，帮助理解裁剪范围：

```cpp
if (_showBoundingBox) {
    // 用 instanced 线框着色器画 10000 个绿色立方体线框
    _lineInstancedShader->use();
    _lineInstancedShader->setUniformVec3("material.color", glm::vec3(0, 1, 0));
    _instancedAsternoids->draw();  // 每个实例画一个 AABB 线框
}
```

### 4.2 相机控制

WASD 移动，鼠标旋转，自由飞行观察裁剪效果：

```cpp
// _cameraMoveSpeed = 10.0f
// _cameraRotateSpeed = 0.05f
```

---

## 5. 动手实操

### 5.1 编译运行

```powershell
cmake --build build --target bonus2 --config Release
.\build\bin\Release\bonus2.exe
```

### 5.2 实验

1. **切换 CPU / GPU 模式**：ImGui 里切换，观察 `drawAsternoidCount` 的变化
2. **开启包围盒显示**：绿色线框 = 每个小行星的 AABB，只有和视锥体相交的才画
3. **飞行探索**：WASD + 鼠标在场景中飞行，看远处的小行星随相机移动被裁剪
4. **开/关 Indirect Draw**：对比传统 draw call 和间接绘制的性能差异

### 5.3 核心 TODO

`frustum_culling.vert` 里的 GPU 裁剪逻辑。思路：
```
对 6 个平面依次测试：
  对 AABB 每个顶点（变换到世界空间后）：
    计算 signedDistance = dot(normal, point) + distance
  如果所有 8 个顶点 signedDistance 都 < 0（全在背面）→ 丢弃
```

---

## 6. 知识延伸

- **Hierarchical Z-Buffer Culling**：用 z-buffer 的金字塔加速遮挡剔除
- **Bounding Volume Hierarchy**：更紧凑的包围体（OBB、包围球）
- **Spatial Partitioning**：八叉树、BSP 树预先划分空间，裁剪从 O(n) 降到 O(log n)
