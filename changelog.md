## TODO
### 20260321
写一个触发大量bank conflict的代码，并nsys查看效果

## changelog

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