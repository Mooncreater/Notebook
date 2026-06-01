---	
comments : true	
---	
	
# Texture Mapping	
	
定义：纹理映射（贴图），将图像映射到三维物体表面以增加表面细节。	
	
!!! tip "核心要点"	
    纹理映射的本质是**查表**：根据 UV 坐标从纹理图像中取样颜色。过滤方式决定了取样质量。	
	
## 动机	
	
用光照反射模型（Phong 等）很难构造出复杂的表面细节（木纹、砖墙、皮肤等）。	
	
:arrow_right: 将预先准备好的**纹理图像**贴到物体表面	
	
## 1. 纹理坐标（UV 坐标）	
	
将纹理空间的坐标映射到物体表面：	
	
- $(u, v) \in [0, 1]^2$ 表示纹理空间坐标	
- 每个顶点除了空间位置，还附带一组 $(u, v)$ 坐标	
- 三角形内部通过**重心坐标插值**得到 $(u, v)$	
	
## 2. 纹理过滤（Texture Filtering）	
	
当纹理分辨率与屏幕像素不匹配时，需要过滤：	
	
### Nearest Neighbor（最近邻）	
	
取最近的纹素（texel），速度快但锯齿明显。	
	
### Bilinear（双线性插值）	
	
取周围 4 个纹素进行双线性插值：	
	
$$c = lerp(lerp(c_{00}, c_{10}, u), lerp(c_{01}, c_{11}, u), v)$$	
	
比最近邻平滑，GPU 默认选项。	
	
### Trilinear（三线性插值）	
	
在 Mipmap 两层之间再加一层插值，消除 Mipmap 层级跳变。	
	
### Anisotropic（各向异性过滤）	
	
当表面倾斜角度大时（如地板远处），在不同方向采用不同采样率，减少模糊。	
	
!!! tip "选择建议"	
    一般场景用 Bilinear，开放世界地面用 Anisotropic 2x-4x，UI 用 Nearest。	
	
## 3. Mipmap	
	
**多级渐远纹理**：预先生成一系列逐步缩小的纹理图。	
	
- 第 0 级：原始分辨率	
- 第 1 级：$1/2$ 分辨率	
- 第 2 级：$1/4$ 分辨率 ……	
	
GPU 根据像素覆盖的纹理区域大小自动选择合适层级。	
	
额外存储开销：$\frac{1}{4} + \frac{1}{16} + \cdots \approx \frac{1}{3}$（仅多 $1/3$）	
	
!!! warning "常见误区"	
    Mipmap 只解决**欠采样**问题（纹理比屏幕分辨率高），不解决放大问题。	
	
## 4. 纹理环绕模式（Wrapping Mode）	
	
当 UV 坐标超出 $[0, 1]$ 时的处理方式：	
	
- **Repeat（重复）**：平铺纹理（常用于地砖、墙壁）	
- **Mirror（镜像）**：镜像平铺	
- **Clamp（钳位）**：将超出部分的 UV 限制到边界值	
- **Border（边框）**：超出部分填充指定颜色	
	
## 5. 凹凸贴图与法线贴图	
	
### Bump Mapping（凹凸贴图）	
	
通过扰动表面法线来模拟凹凸效果，不改变几何体。	
	
- 存储高度偏移 $h(u, v)$	
- 计算偏导 $\frac{\partial h}{\partial u}$、$\frac{\partial h}{\partial v}$	
- 扰动法线：$\mathbf{N'} = \mathbf{N} + \Delta\mathbf{N}$	
	
### Normal Mapping（法线贴图）	
	
直接存储扰动后的法线方向（RGB 对应 XYZ），效果更精确，是 Bump Mapping 的进阶。	
	
### Displacement Mapping（位移贴图）	
	
真实改变几何体顶点位置（与 Bump/Normal 仅改变法线不同），效果最准确但开销最大。	
	
!!! tip "对比"	
    Bump/Normal：看起来有凹凸，轮廓线仍是平的。Displacement：真有凹凸，轮廓线也会变化。	
	
## 6. 环境贴图（Environment Mapping）	
	
用纹理模拟周围环境对物体的反射：	
	
- **Cube Map**：六面立方体纹理，GPU 原生支持	
- 常用于反射、折射的快速近似