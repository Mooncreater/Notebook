---	
comments : true	
---	
	
# 光照	
	
## Radiosity 算法	
	
Radiosity 算法是一种全局光照算法，模拟场景中漫反射面之间的能量传递。与光线追踪不同，Radiosity 主要处理**漫反射-漫反射**之间的光照交互。	
	
### 1. 基本思想	
	
- 将场景表面离散化为多个 **patch**	
- 每个 patch 既是光的接收者也是发射者	
- 通过求解能量平衡方程计算每个 patch 的最终辐射度	
	
### 2. Radiosity 方程	
	
单个 patch $i$ 的辐射度 $B_i$：	
	
$$B_i = E_i + \rho_i \sum_{j} B_j F_{ij}$$	
	
- $E_i$：patch $i$ 的自发光	
- $\rho_i$：patch $i$ 的反射率	
- $F_{ij}$：**Form Factor**（形状因子），表示从 patch $j$ 到达 patch $i$ 的能量比例	
	
### 3. Form Factor	
	
$F_{ij}$ 表示离开 patch $i$ 的能量中到达 patch $j$ 的比例：	
	
$$F_{ij} = \frac{1}{A_i} \int_{A_i} \int_{A_j} \frac{\cos\theta_i \cos\theta_j}{\pi r^2} V_{ij} \, dA_j \, dA_i$$	
	
- $V_{ij}$：可见性函数（两点之间是否有遮挡）	
- $r$：两 patch 之间的距离	
	
**性质**：	
- 互易关系：$A_i F_{ij} = A_j F_{ji}$	
- 封闭性：$\sum_j F_{ij} = 1$	
	
### 4. 求解方法	
	
#### Progressive Refinement（逐步求精）	
	
不直接求解大型线性方程组，而是通过迭代逐步"射击"能量：	
	
1. 选择当前未发射能量最多的 patch	
2. 将该 patch 的能量"射击"到所有其他 patch	
3. 更新各 patch 的辐射度	
4. 重复直到收敛	
	
:arrow_right: 优点：可以逐步看到渲染结果，适合交互式应用	
	
### 5. 特点	
	
- ✅ 视角无关：计算完成后可从任意角度观察	
- ✅ 完美处理漫反射间的颜色渗透（color bleeding）	
- ❌ 难以处理镜面反射和折射	
- ❌ 场景网格化开销大	
	
## 光线追踪算法	
	
光线追踪（Ray Tracing）通过**逆向**追踪光线来模拟真实光照：从视点出发，穿过像素发射光线，追踪其与场景中物体的交互。	
	
概念：模拟光的物理行为——反射、折射、阴影，产生高真实感图像。	
	
### 1. 基本流程	
	
1. **Ray Generation**：从视点穿过每个像素发射一条主光线（primary ray）	
2. **Ray-Object Intersection**：计算光线与场景物体的最近交点	
3. **Shading**：在交点处计算颜色（使用 Phong 等局部光照模型）	
4. **递归追踪**：从交点发射反射光线 / 折射光线，递归计算颜色	
	
### 2. 光线-物体求交	
	
#### 光线方程	
	
$$\mathbf{P}(t) = \mathbf{O} + t\mathbf{D}, \quad t > 0$$	
	
#### 与球体求交	
	
球体方程：$|\mathbf{P} - \mathbf{C}|^2 - R^2 = 0$	
	
代入光线方程解二次方程：	
	
$$at^2 + bt + c = 0$$	
	
其中 $a = \mathbf{D} \cdot \mathbf{D}$，$b = 2\mathbf{D} \cdot (\mathbf{O} - \mathbf{C})$，$c = |\mathbf{O} - \mathbf{C}|^2 - R^2$	
	
- 判别式 $\Delta < 0$：不相交	
- $\Delta = 0$：相切	
- $\Delta > 0$：两个交点，取较小的 $t > 0$	
	
#### 与三角形求交	
	
使用重心坐标表示交点：	
	
$$\mathbf{P} = \alpha \mathbf{A} + \beta \mathbf{B} + \gamma \mathbf{C}, \quad \alpha + \beta + \gamma = 1$$	
	
:arrow_right: 常用 **Möller–Trumbore 算法**直接求解	
	
### 3. Whitted 风格光线追踪	
	
在每个交点处计算三种光线：	
	
- **反射光线**：$\mathbf{R} = \mathbf{I} - 2(\mathbf{I} \cdot \mathbf{N})\mathbf{N}$	
- **折射光线**：由 Snell 定律 $n_1 \sin\theta_1 = n_2 \sin\theta_2$ 决定	
- **阴影光线**：向光源发射，判断该点是否处于阴影中	
	
最终颜色：	
	
$$I = I_{local} + k_r I_{reflect} + k_t I_{refract}$$	
	
### 4. 加速结构	
	
朴素求交遍历所有物体开销太大 :arrow_right: 使用空间加速结构：	
	
- **BVH（Bounding Volume Hierarchy）**：用包围盒层次树组织物体	
- **KD-Tree**：用轴对齐平面对空间进行二分	
	
### 5. 特点	
	
- ✅ 真实感高：自然地处理反射、折射、阴影	
- ❌ 计算量大，早期无法实时	
- ❌ 只能处理从视点可见的路径（难以模拟焦散等）	
	
## 阴影	
	
### 1. 阴影光线（Shadow Ray）	
	
在光线追踪中，从表面交点向光源发射一条**阴影光线**：	
	
- 如果在到达光源之前与任何物体相交 :arrow_right: 该点处于阴影中	
- 否则 :arrow_right: 该点被该光源照亮	
	
$$I = I_{ambient} + \sum_{light} V_i \cdot I_{light_i}$$	
	
其中 $V_i \in \{0, 1\}$ 为可见性（shadow ray 测试结果）	
	
### 2. 阴影分类	
	
- **硬阴影（Hard Shadow）**：由点光源产生，边缘清晰	
- **软阴影（Soft Shadow）**：由面光源产生，包含**本影（umbra）**和**半影（penumbra）**	
	
### 3. Shadow Mapping	
	
一种光栅化中的实时阴影技术：	
	
1. **Pass 1**：从光源视角渲染场景，存储深度信息到 shadow map	
2. **Pass 2**：从相机视角渲染，对每个片段：	

   - 将其转换到光源空间	
   - 比较当前深度与 shadow map 中存储的深度	
   - 深度更大 :arrow_right: 在阴影中	
	
⚠️ 问题：阴影锯齿（shadow acne）、分辨率不足	
:arrow_right: 改进：PCF（Percentage Closer Filtering）、CSM（Cascaded Shadow Maps）