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


## FiD-KD ICLR 2021 Facebook

[Distilling Knowledge from Reader to Retriever for Question Answering](https://zhuanlan.zhihu.com/p/576171199) 

1. **动机**：fid以及之前的REALM、RAG都没有对检索到的结果进行监督。作者注意到fid将所有encoding拼接后，会进行cross attention，认为其中的attention的值的大小是一个很好的监督信号，可以根据此对检索到的文章进行打分（**因为一个文本段中的token被关注的越多，就越可能是回答该问题的有效参考段落**）。

2. **做法**：使用sequence-to-sequence模型作为阅读器，并使用decoder对输入文档的注意得分作为标签来训练检索器

   - **<u>（1）怎么计算检索器标签</u>**：给定问题q和对应的支持文档集 $D_q=(p_k)_{1<=k<=n}$ ，取decoder中**每一层**的**每一个注意力头**的**第零个token**的**query向量Q，**与**encoder拼接序列**上的**某一个检索段落** $p_k$对应的**每一个token**的**key向量K，**计算**Q与K**之间的点积分数，对所有分数取平均，作为检索段落$p_k$的注意力得分$G_{q,p_k}$ 。

   - **<u>（2）怎么蒸馏cross-attention分数到bi-encoder</u>**：最小化retriever打分和cross-attention score分布之间的kl散度：

    ![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/e4f9e8db-d468-4e4b-8ce4-fa894b1c275e)

   - **<u>（3）怎么迭代训练</u>**：首先，对于每个q，会得到一个初始的support documents $D_q^0$，采样迭代过程，每次迭代包含下面4个步骤。

     1. **使用每个q和其对应的$D_q^0$ 训练生成器  G**，注意，在每一次新的训练迭代中，这一步的生成器都从T5基础上重新初始化，防止被前轮不好的检索结果带偏。
     2. 使用学习好的生成器 G，**获取聚合后的注意力分数 $(G_{q,p})_{q \in Q, p \in D_q^0}$**
     3. 用得到的注意力分数，通过kl散度的方式，**训练检索器 R**
     4. 用学习好的检索器 R 重新**为每个q召回对应的support documents** 

   - 注意，迭代训练中的初始support docs非常重要，可以使用 bm25 或者事先训练好的dpr

## RETRO 2022 Deepmind

[Improving Language Models by Retrieving from Trillions of Tokens](http://jalammar.github.io/illustrated-retrieval-transformer/) 


几个takeaway是：1）不训练检索器就很好使，RETRO直接用冻结的BERT作为检索器，效果也很好。2）cross attention固然好用，像RETRO这样设计新的深度融合可能会更好。

**RETRO相比REALM，采用的是chunks维度的检索**。

1. **动机**：通过检索增强的方式增强小的语言模型，从而达到和大规模语言模型相媲美的性能。

   - 预测下一个单词——本质上是在句子末尾填空。

   - 认为**将language information和world knowledge分开很重要**（填空需要的东西，有的仅仅依靠于语言信息就可以填出来），用语言模型编码language information很合理，但是对于factual和world-knowledge information来说效率很低。 ===> 在LM中引入retrieval method

    ![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/4b795796-b823-4b63-841b-a3d7877d14c7)


2. **retrieval database**：k-v存储，k是bert编码的sent emb，v有2个部分：（1）neighbour，计算出key的文本；（2）completion，原始文本的续写。

   ![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/3cf2a83a-bc6e-42fe-8178-fc2a1aaa2f69)


3. **The Database Lookup**：

   ![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/19823af8-016c-4546-8898-3299cb425011)


4. **RETRO Architecture**：

   - **encoder**：检索到的neighbors，作为输入，编码得到keys和values，送往decoder。
   - **decoder包含两类**：**Standard decoder block** (ATTN + FFNN)；**RETRO decoder block** (ATTN + Chunked cross attention (CCA) + FFNN)；从第9个块开始，每3个块遇到一个retro decoder block，所以第 9、12、15…32 层是 RETRO 块。

   ![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/7cb47da3-bfe8-4673-acdd-fc207048f439)


   - decoder的流程
     - **Decoder blocks**：It applies self-attention on the prompt token (causally, so only attending to previous tokens), then passes through a FFNN layer.
     - **RETRO decoder**：合并检索到的信息

5. overall

   - 每个训练 sequence 先被切分为 chunks, 分别用它们在database中检索到的k近邻增强。
   - Chunked cross attention (CCA) 可以看下面的示意图，其中，原本长度为n的序列，被切成l个块，每个块m个词。$H_u^+$  保留了块 $C_u$ 中的最后一个token的emb和 下一个块 $C_{u+1}$  中的前m-1个token emb。
   - 注，块 $C_u$ 中的last token是第一个能访问该块检索到的内容 $E_u$ 的token，保证了likelihood中的autoregressive性。
   - 在 $H_u^+$ 和 $E_u$ （块$C_u$ 检索回来的embs）之间计算cross- attention，将得到的emb再替换回去。

![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/2f88a4f8-fcc3-45cc-841c-3d6b78d972b5)

![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/78489799-e187-4c7e-8a9e-09096f565f95)


**likelihood 的定义也是 autoregressive 的** : the probability of the 𝑖-th token of the 𝑢-th chunk $x_{(u-1)m+i}$ , only depends on previously seen tokens  $(x_j)_{j<(u-1)m+i}$ and on the data retrieved from the previous chunks $(RET_D(C_{u'}))_{u'<u}$


## REPLUG 2023 meta

[Retrieval-Augmented Black-Box Language Models, 2023](https://blog.csdn.net/qq_52852138/article/details/130775281) 

1. 动机：

   - 将语言模型视作黑盒，只需要将检索到的文档拼到原有输入前面，不需要更新大模型参数。在该架构中，通过更新检索器来提升性能。

2. REPLUG (inference) 做法：

   - 给定查询x，先用检索器检索出top-k个文档集合 D'
   - 将每个文档和x拼接，并行送给大模型
   - 预测下一个词y的概率由加权平均决定，$\lambda(d,x)$ 是检索器打分在D‘上softmax后的结果。 

   $$
   p(y|x,D')=\sum_{d\in D'} p(y|d \circ x) \cdot \lambda(d,x)
   $$

3. REPLUG LSR: 用语言模型反馈的监督信号，调整检索器。（一个文档如果对大模型越有帮助，越应该被检索回来）

   - 给定q，还有检索到的top-k个文档，对2个分布计算kl散度。最小化损失函数来优化检索器，LM保持不动。
   - 第一个分布：检索器对这k个文档的打分，过softmax
   - 第二个分布，生成器对这k个文档的打分，即，输入d和x，生成ground truth y的概率，同样过softmax
   - 注：因为检索器参数在训练过程中更新，参数更新后document embedding会变化，因此**每隔T步就重新算一次document embedding**，并重复上述过程。

## inpairs sigir 2022

[Data Augmentation for Information Retrieval using Large Language Models](https://arxiv.org/pdf/2202.05144.pdf) 

1. 方法：

   - 对于一个文档，前面加3组 q-d pair 构建instruction，然后用LLM生成query，以及对应的生成概率。（无论输入文档d如何，无论是哪个数据集的，前缀3组 q-d pair始终相同），提供了两种模板。

   - 基于这个方法，就可以从文档集合中随机采样一些文档，对每个文档生成一个对应的query，最终构成训练的正样本。用于微调精排模型。
   - 不执行任何预训练来使模型适应目标语料库

   
![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/34bf1ece-0e17-461a-afb6-65135a9554f3)

![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/39730649-84b9-44e9-81f3-113cbad93cc6)


## PROMPTAGATOR iclr 2023 google

[Few-shot Dense Retrieval From 8 Examples](https://zhuanlan.zhihu.com/p/585269408) 

1. 动机：关注few-shot dense retrieval，设置每个任务只有一个简短的描述和少量的examples，用大语言模型基于prompt进行数据生成，用于检索器训练。

   - 和 Inpairs 的区别：1. prompt 有task描述信息；2. 小的retriever或者ranker也可以达到很好的效果

2. 做法：

   - 【prompt-base <u>**query generation**</u>】把已有的q-d对作为prompt，给不同的d，让大模型生成对应的q，获得更多的q-d对数据。
   - 【consistency filtering using only generated data 】用生成的数据训练初始的检索器，而后对生成的数据进行过滤：给定生成的(q,d)对，用刚刚训练好的检索器对q进行检索，如果对应的d出现是其top1，则保留该生成的(q,d) pair
   - 【few-shot promptagator retriever】分为pre-train和fine-tune
     - pre-train：用T5的encoder作为dual-encoder的初始化，在c4数据上预训练，预训练方法参见contriever（同一个文档的两个随机的段落作为正样本对）。
     - fine-tune：在生成的q-d数据上微调，同样使用in-batch random negatives。在训练了几个epoch后，对生成数据进行过滤。用过滤后的数据继续微调dual-encoder。


## Hypothetical Document Embedding (HyDE) 2022 cmu

Precise Zero-Shot Dense Retrieval without Relevance Labels

- 给定一个query，首先用zero-shot的方式指示InstructGPT生成一个能回答该query的假设文档，使用无监督对比学习Contriever把doc编码成embedding vector，然后从真实的语料库中找到假设文档最近邻相似的文档。
- 把dense retrieval任务分解为了两个任务，生成任务 + doc-doc相似性比较任务。NLG and NLU ，替换掉了显式的相关性建模

![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/c56336cc-1f65-43b7-9c7a-dba87dca78ed)


## GenRead (generate-then-read) iclr 2023 

[Generate rather than Retrieve: Large Language Models are Strong Context Generators](https://openreview.net/pdf?id=fB0hRu9GZUS) 

![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/006eabad-fe65-4fee-a70f-8c713e0b44a0)


1. **zero-shot做法**：no training data – neither questions nor contextual documents.
   - **Step1 generate**：用instructGPT，对给定的question生成documents
   - **Step2 read**：使用生成的句子d和查询一起作为输入，送给大模型产生answer
2. **supervised 做法：**保证了多样性～
   - **step 1 generate one initial document per question**：问题库Q（q1,q2...），对每个q 用大模型生成或者使用 BM25 从维基百科检索出一个doc，得到q-d pair set $\{q_i, d_i\}_{i=1}^{|Q|}$
   - **step 2 encode each document，do k-means clustering**：使用大模型（eg，GPT3）对每个q-d pair编码。而后对|Q|个编码得到的向量进行k-means聚到k类，k是一个变量。
   - **step 3 sample and generate k documenrs**：从每个类中采样n个q-d pair， n是一个超参数。每个类下的n个q-d pair，作为in-context demonstrations，对于新输入的query，分别和不同类下的n个q-d对组成输入，让大模型为query生成不同类别下的doc。
   - 使用类似fid的方式，把q和生成的不同类别的doc，送给decoder，得到answer。

![image](https://github.com/sunxiaojie99/retrieval-augmented-generation/assets/41667783/f4bad59c-e26c-459e-94aa-ae023fa1297c)



