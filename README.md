# retrieval-augmented-generation

当我们使用LM时候，我们到底允不允许它“拖家带口“，即训练完我们必须要把所有的训练数据扔掉，即**“扔掉书本，闭卷考试“**，还是说允许在预测时候允许带着所有训练数据，即**“允许带着书进入考场，开卷考试”**。

这两种方式不仅仅是持续学习角度里面的不同的setting，从我们做NLP的角度来讲，实际的工业部署需求是一定要考虑在其中的，到底是<u>**把知识存到模型**</u>里面效率高一点还是**<u>把知识放在内存里面检索</u>**出来效率高，随着软硬件结构的改变随时洗牌完全有可能。

[检索、提示：检索增强的（Retrieval Augmented）自然语言处理](https://zhuanlan.zhihu.com/p/470784563) 

## kNN-LMs ICLR 2020

Generalization through Memorization: Nearest Neighbor Language Models

[ref](https://zhuanlan.zhihu.com/p/90890672?utm_id=0) ：没有修改神经网络模型，仅仅是将LM 和knn检索结果插值。

1. **背景知识：**knn思想：给定新样本，在已有样本中检索和其距离最小的k个样本，认为这k个样本中出现频率最高的类别就是该新样本的类别。
2. **动机**：传统用语言模型给定 $context_i$ （$w_1, w_2, ..., w_{i-1}$） 去预测下一个词 $w_i$，认为传统的autoregressive预测的方式，很难充分建立长距离的依赖。

3. **做法：**结合训练集中 上文向量表示 的k最近邻和当前语言模型的预测概率
   - 根据训练集建立一个datastore，key是不同的前文编码后的表示，value是该前文对应的下一个词。
   - 当给定一个前文时，先用语言模型对此进行编码，使用此向量利用faiss去执行knn，检索k最近邻的前文向量，得到其对应的下一个词，形成 $P_{KNN}(y|x)$ 
   - 把这个概率和语言模型的概率结合起来，作为生成下一个词的指导分布：$P(y|x)=\lambda P_{KNN}(y|x) + (1-\lambda) P_{LM}(y|x)$

4. **问题**：训练集中的每个token都需要自己作为value，其前文作为key，检索的数据规模相当大，虽然用faiss加速，依旧是一个问题。



<img src="https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/32ec3636-0fa7-4be0-86c8-b7c0ce7ee369" alt="image-20230622202242596" style="zoom:30%;" />


