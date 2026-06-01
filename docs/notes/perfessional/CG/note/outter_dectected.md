---	
comments : true	
---	
	
# 剔除与裁剪	
	
剔除（Culling）和裁剪（Clipping）是渲染管线中用于减少不必要的计算、提高性能的关键步骤。目的是尽早丢弃不可见或不在视野内的几何体。	
	
!!! tip "核心要点"	
    剔除在**物体级别**丢弃不可见面，裁剪在**图元级别**切除超出屏幕的部分。先剔除再裁剪，可以省去大量无用计算。	
	
## 1. 背面剔除（Back-Face Culling）	
	
对于封闭物体，背对相机的面永远不可见，可以直接丢弃。	
	
### 判定方法	
	
根据多边形法线 $\mathbf{N}$ 与视线方向 $\mathbf{V}$ 的夹角：	
	

$$\mathbf{N} \cdot \mathbf{V} > 0 \quad \Rightarrow \quad \text{背面，剔除}$$	
	
- 约减少 **50%** 的面片处理量	
- 在现代 GPU 中由硬件自动完成	
	
!!! tip "注意"	
    只有**封闭物体**的背面才不可见。渲染树叶、布料等双面材质时需要关闭背面剔除。	
	
## 2. 视锥体裁剪（View Frustum Culling）	
	
只有位于**视锥体**（View Frustum）内的物体才可见。	
	
视锥体由 6 个平面定义：	
- 近平面（Near） / 远平面（Far）	
- 左平面 / 右平面	
- 上平面 / 下平面	
	
### 方法	
	
用包围体（AABB / Bounding Sphere）与视锥体做相交测试：	
- 完全在外 :arrow_right: 剔除	
- 完全在内 :arrow_right: 保留	
- 部分相交 :arrow_right: 保留（后续裁剪）	
	
## 3. 直线裁剪算法	
	
当图元部分超出屏幕边界时，需要裁剪到可见区域。	
	
### Cohen-Sutherland 算法	
	
为平面区域 9 个分区编码（4 位区域码）：	
- Bit 0：在左边框之左	
- Bit 1：在右边框之右	
- Bit 2：在下边框之下	
- Bit 3：在上边框之上	
	
对线段两端编码：	
- 两端 code 均为 0 :arrow_right: 线段完全在内部，保留	
- 两端 code 按位与 $\neq$ 0 :arrow_right: 线段完全在外部，丢弃	
- 否则 :arrow_right: 求线段与边框交点，递归裁剪	
	
### Liang-Barsky 算法	
	
用参数方程表示线段，通过不等式约束直接计算交点：	
	

$$\begin{cases} x_{min} \leq x_1 + t\Delta x \leq x_{max} \\ y_{min} \leq y_1 + t\Delta y \leq y_{max} \end{cases}$$	
	
比 Cohen-Sutherland 更高效，减少浮点除法。	
	
## 4. 多边形裁剪	
	
### Sutherland-Hodgman 算法	
	
依次用每个裁剪边对多边形进行裁剪：	
	
1. 维护输入/输出顶点列表	
2. 对于每条裁剪边，顺序处理多边形每条边	
3. 根据顶点在边的内外决定是否输出	
	
适用于凸多边形裁剪窗口。	
	
## 5. 遮挡剔除（Occlusion Culling）	
	
移除被其他物体完全遮挡的物体，避免渲染不可见内容。	
	
常用方法：	
- **Occlusion Query**：GPU 硬件查询，渲染包围体后检查可见像素数	
- **PVS（Potentially Visible Set）**：预计算静态场景的可见性集合	
- **Hierarchical Z-Buffer**：层次深度缓冲，快速拒绝被遮挡物体	
	
!!! warning "性能注意"	
    遮挡剔除本身也有开销。对于简单场景，剔除开销可能大于渲染节省。通常只在复杂场景（室内、城市场景）中使用。	
	
---	
	
### 剔除与裁剪总结	
	
| 类型 | 阶段 | 剔除标准 | 开销 |	
|------|------|----------|------|	
| 背面剔除 | 图元装配前 | $\mathbf{N} \cdot \mathbf{V} > 0$ | 极低（硬件） |	
| 视锥体裁剪 | CPU / GPU | 包围体 vs. 视锥体 | 低 |	
| 直线裁剪 | 光栅化前 | 屏幕边界 | 低 |	
| 遮挡剔除 | 渲染前 | 被前方物体遮挡 | 中 |