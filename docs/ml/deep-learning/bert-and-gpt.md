# BERT & GPT

## 参考

* [BERT 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1PL411M7eQ)
* [【機器學習2021】自督導式學習 (Self-supervised Learning) (二) – BERT簡介](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=19)
* [【機器學習2021】自督導式學習 (Self-supervised Learning) (三) – BERT的奇聞軼事](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=20)

## 论文

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)：
    * 提出 BERT 模型，基于 Transformer 的编码器，双向编码
    * 提出预训练的方法；将预训练+精调引入 NLP，可用于解决众多问题
    * 将具有亿数量级参数的模型引入 NLP，引发一轮暴力竞赛
    * 由于舍弃解码器，不适合机器翻译、摘要提取等生成式任务
