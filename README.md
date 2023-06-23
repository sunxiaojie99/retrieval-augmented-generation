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

## **REALM** 增强开始 ICML2020 google

[Retrieval-Augmented Language Model Pre-Training](https://blog.csdn.net/Forlogen/article/details/104343229) ：通过retrieve-then-predict的方式，统一了pre-train和fine-tune的过程，检索模型和预训练模型共同训练。

1. **背景**：之前预训练模型的pre-train和finetune，模型都是计算条件概率 P(y|x)。在pre-train时，x是被mask处理过后的文本，y为被mask的部分；fine-tune的时候，x是question，y是对应的answer。

2. **动机**：完全依赖于模型权重来学习知识是一件非常低效的事情，正确的做法是让模型 学会找 + 理解。

   - **通过retrieve-then-predict进行拆解**，P(y|x)分为p(z|x)和p(y|z,x)。

   - 第一个是**知识检索模型** neural knowledge retriever，建模 P(z=辅助性知识|x=query)；第二个是**知识增强编码器** knowledge-augmented encoder，建模 P(y=answer | z,x)
   
     ![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/8f215d02-bf95-4468-b076-aca046c2cacc)

3. **做法**：在预训练和finetune阶段都添加一个 **知识检索**（knowledge retrieval） 的步骤，检索模型和预训练模型是共同训练的。

   - **pre-training**：采用和BERT一致的MLM的策略进行建模，因此模型需要预测每个被MASK的内容。进行mlm预测前会先通过检索模型检索出相关的文档。
   - **fine-tuning：** 这里解决的Open-QA任务，模型同样先从corpus中检索出相关的文档，利用检索结果完成open-qa任务。
   
![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/201f84ff-7638-486c-9a22-486117a7ea2f)


4. 实现：
   - 在pre-train和fine-tune的时候，都可以异步更新retrieval的索引，作者在pre-train的时候进行了此操作，即，在训练几百步后再异步的更新表示向量所对应的索引。在fine-tune的时候，为了简化，固定使用了预训练模型编码好的corpus emb，但是依然会更新retrieval ，获取最新的query emb


## RAG，通用增强模型 NeurIPS 2020

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](在训练过程中，联合训练检索器和生成器，) 

1. 动机：之前的REALM 是基于MLM任务的，只方便做提取任务，这里用generator 增大应用面。**只做了微调，没做预训练，REALM是预训练和微调都做的。**

2. 做法：

   - 同样是先用x检索z，再用x和z作为前文去生成y 。给出了两种不同的设置 **RAG-sequence和RAG-token**，分别是（1） 先乘后加，使用相同的检索文档来生成完整的序列 ，把不同文档z的概率加起来；（2）先加后乘，给定一个token，先用全部文档z都去生成这个token，每个token的最终概率是该文档的检索概率 \* 基于该文档生成这个token的概率。最后把y中所有token的概率乘起来。

   - retriever 用的dpr；generate用的 bart-large。

     ![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/a5dd9917-49c0-4eae-8ea6-b468af1444c9)


     ![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/f5473ae6-bcb0-4c26-963a-e759aeb28b5d)

3. 实现：

   - 在微调过程中，联合训练检索器和生成器，采样上面的边际似然函数进行训练，作者不认为定期更新doc emb有必要，只微调了查询编码器Eq和生成器。


## fid EACL2021 简单高效的Generator改进

[Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://aclanthology.org/2021.eacl-main.74.pdf) 

https://github.com/facebookresearch/FiD

1. **动机**：认为与检索模型相比，生成模型非常善于将多个passage的信息合成：
2. **做法：**详细的公式可以见fid-kd中的
   - 很简单直接，将检索到的每个doc都分别和query一起通过encoder进行编码，得到多个编码向量，拼接到一起，输入decoder生成最终的回复。所以，叫做fusion-in-decoder。（作为encoder得到的k和v，进行cross- attention）
   - 形式化：（1）给定query $q$ 和一个支撑passage，使用 `question:` ，`title:` ，`context:` 添加在query，wiki title和每个passage的text前，送入编码器编码。把所有passage经过处理的编码拼接成全局表示 $X$，其维度是$\sum_k(l_k)\times d$，$l_k$ 是第k个passage对应得到的token个数。（2）而后，decoder会执行常规的autoregressive过程，即：self-attention，cross-attention和feed-forward 模块。注意，只有cross- attention模块会显式用刚刚得到的$X$ 作为输入，作为attention中的key和value。

![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/3120174a-d28a-42b9-8b70-757b7c5cf996)


