## TODO
### 20260321
写一个触发大量bank conflict的代码，并nsys查看效果

## changelog

### 20260326
1. 删除ai生成的flash_attn代码。
2. 增加flash attn笔记，增加对特性理解。
3. ToyFlashAttention开始框架编码，暂定目标先通过pytorch实现算子并接入跑通。

### 20260325
读完vllm/v1/attention/backends/flash_attn.py，了解Flash Attention框架，以及主要优化方式cuda计算图，cascade/prefix公共前缀合并计算，fp8量化scale & descale, sink, GQA DCP等。

### 20260324
阅读vllm/v1/attention/backends/flash_attn.py, 理解FlashAttentionMetadata，FlashAttentionBackend的用途。后续考虑废弃ai生成版本，重新手写。

### 20260323
1. matmul gpu tile版本初步跑通，待进一步优化，以及和标准方式对比。
2. flash attention cpu 版本reference就绪。
3. flash attention backend 框架ai生成，vllm可以识别，但不能跑。

### 20260322
matmul gpu tile版本有逻辑错误，待确认正确方式。

### 20260321
matmul cpu朴素 & cpu转置后完成。

### 20260320
transpose朴素gpu版本 & shared_memory tile版本 & bank conflict solve版本

### 20260319
完成online softmax gpu版本，未优化。

### offline
完成histogram & reduction sum & softmax gpu