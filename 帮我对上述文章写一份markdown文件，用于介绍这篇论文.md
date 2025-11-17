# **TSP-Former: 一种物候引导的Transformer烟草识别模型**

**论文标题:** TSP-Former: A Phenology-Guided Transformer for Tobacco Mapping Using Satellite Image Time Series

**作者:** Huaming Gao, Yongqing Bai, et al.

## **摘要**

精准、及时地获取烟草种植分布信息对于农业规划和公共卫生管理至关重要。然而，传统遥感方法常常因不同作物间光谱特征相似以及种植模式的地域差异而受到限制，尤其是在模型的跨区域应用上。为解决这一挑战，本研究提出了一种新颖的、由物候引导的深度学习模型——**TSP-Former**，旨在利用卫星影像时间序列（SITS）精准识别烟草。

## **核心问题**

传统的烟草遥感识别面临两大难题：

1. **同物异谱/异物同谱**：烟草在特定生长期与玉米等其他作物的光谱特征非常相似，难以区分。  
2. **泛化能力弱**：由于不同地区的种植时间、作物类型和生长环境存在差异，在一个区域训练的模型很难直接应用于其他区域。

## **我们的解决方案**

为了克服上述难题，我们提出了一种融合遥感先验知识的深度学习框架。

### **1\. 烟草光谱物候变量 (TSP)**

我们通过分析发现，烟草在快速生长期，其**红边-2 (Red Edge-2)** 波段的反射率增长速率显著高于其他作物。基于这一独特的物候特征，我们构建了一个名为 **TSP (Tobacco Spectral-Phenological)** 的变量。

* **作用**：TSP变量能够有效量化烟草的独特生长模式，显著增强烟草与其他作物（尤其是玉米）的可分性，作为强大的先验知识引导模型学习。

*图：TSP变量可以清晰地突显烟草种植区（暖色调区域）*

### **2\. TSP-Former 模型架构**

基于TSP变量，我们设计了一个全新的Transformer架构——**TSP-Former**。该模型通过两个核心创新模块，将遥感先验知识与深度时空特征进行高效融合。

* **中央先验注意力模块 (CPAM)**：该模块将TSP先验知识与中心像素周围的空间光谱信息进行自适应融合，强化模型对烟草特征的表达能力。  
* **NDVI增强时序解码器 (NDTD)**：利用NDVI时间序列作为权重，使模型在解码时更加关注作物生长的关键物候阶段，从而保留最重要的时序信息。

*图：TSP-Former模型流程图*

## **主要贡献与成果**

1. **卓越的分类精度**：TSP-Former在四个中国主要的烟草种植区进行了验证，平均加权F1分数达到 **0.8781**，总体精度 (OA) 达到 **0.8761**，显著优于随机森林 (Random Forest) 及其他先进的深度学习模型。  
2. **强大的跨区域泛化能力**：通过引入TSP先验知识，模型有效克服了地域差异带来的挑战，在远离训练区的测试点依然保持了高精度，解决了传统模型的泛化难题。  
3. **TSP变量的模型无关价值**：实验证明，TSP变量本身就是一个高效的特征。将其整合到随机森林和STNet等其他模型中，同样能显著提升这些模型的性能和泛化能力，例如在习水县测试区，使随机森林的Kappa系数提升了 **23.88%**。

## **结论**

本研究证实，将作物特有的物候先验知识融入时间序列深度学习模型，是实现大规模、高精度作物遥感制图的有效途径。TSP-Former为异构化和数据受限的农业区提供了一种稳健且可扩展的解决方案。

**如何引用:**

@article{gao2024tspformer,  
  title={TSP-Former: A Phenology-Guided Transformer for Tobacco Mapping Using Satellite Image Time Series},  
  author={Gao, Huaming and Bai, Yongqing and Sun, Qing and Wang, Haoran and Tian, Xiangyu and Ma, Hui and Li, Yixiang and Che, Xianghong and Chen, Zhengchao},  
  journal={Remote Sensing of Environment (Preprint)},  
  year={2024}  
}  
