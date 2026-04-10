## TODO
### 20260321
写一个触发大量bank conflict的代码，并nsys查看效果

## Dev Log
###20260410
1. 增加layout对qk_matmul_reduction 和score_reduction两个临时tile以及accessor, 增加q k全局id到tile内的映射accessor, 完成qk矩阵乘法部分。增加严格的数值检查。
2. flash_attn_func_v3.cu重构了shared memory layout、QK分块归约、非有限值检查，并开始接入online softmax初始化。

### 20260409
开始写第三版本, 这版本开始注意定义入参param struct, tile layout, 定义充足的安全函数，accessor降低心智负担。
1. 完成ParamSet定义，TileLayout定义，并实现一组accessor & 断言，保证访问正确。完成q tile & k tile load

### 20260408
定位执行推理可以得到结果不一致的原因，原因是q_tile每次读取8个q row, k_tile每次读取8个k row,此时处理时滑动窗口出错。退化到每次只处理1个row时，问题消解。但是还存在问题。现阶段排查下来已经遇到的问题包括：
1. 数值精度问题，应该对输入tile 输出tile 矩阵乘用bf16, 对于online softmax部分用fp32，提高精度，实现两阶段精度处理。
2. 精度处理时应该提供一组安全函数，用来提供对有效&无效数据的兼容，代码更清晰。
3. 应该有安全检测helper函数用于精度问题debug
4. 滑动窗口处理需要结合mask重新设计，否则单block无法处理多q row。
5. headdim对小于warpSize以及非2的幂的场景存在兼容问题
综上，暂不打算在当前代码上堆补丁，考虑重新实现。

### 20260407
MileStone: 
1. flash attn cuda代码跑通，计算结果有错误(nan稳定性问题)。
2. 增加调试辅助函数check_nan_val & check_infinite_val, 封装计算，排查nan引入。
3. 增加safe_sub & softmax_div避免减法引入nan。
4. head_dim=64 num_head=2 token=6 对拍测试初步通过。执行推理可以得到结果不一致，怀疑是数值精度问题。
torch结果:
> I am a former student of the University of California, Berkeley, and I am a member of the Board of Trustees of the University of California
Toy Attention cuda结果
> The number of the number of the number of the number of the number the number of the number of the number the of number the number the number the


### 20260406
排除flash attn cuda实现的大多数边界问题，还缺少head dim对非2指数幂支持，以及稳定性提升。

### 20260403
完成一版本单kernel的flash attention代码开发。有bug,待调试排查。

### 20260402
梳理Q @ K^T和softmax部分逻辑错误，跑通进度60%

### 20260401
1. 完成q & k转换，tile gather。
2. 确认当前版本的block划分维度，在架构层面存在阻塞。当前是以QK^T的输出矩阵维度划分的tile, 不利于online softmax实现，导致需要block间归约，滑动窗口也不便于提前到QK^T前进行和tile内。准备弃用当前实现，重构。
3. 新框架以thread.x作为一个head_dim, threadIdx.y作为对Q的token seq的chunk划分。完成Q @ K^T 和softmax的框架，有bug未排查。

### 20260331
cuda attention kernel 完成框架，内部matmul & online softmax初步雏形， 还缺少地址转换，最后一步P @ V 以及TILE间sum & m归约。

### 20260330
MileStone: 
1. 完善backend, 因为GPT需要GQA支持，切换base model为gpt2后已跑通。
2. cuda算子torch框架接入完成, 增加对拍测试代码。

### 20260329
ai生成了impl和meta实现，backend待接入，未跑通。

### 20260327
MileStone: 
1. 完成flash_attn_varlen_func参数含义梳理。
2. 完成ToyFlashAttention的flash_attn_varlen_func_without_block的实现，并让ai生成对拍单测，运行通过。
3. 完成ToyFlashAttention的flash_attn_varlen_func_with_block的实现，并让ai生成对拍单测，测试通过。

以上测试基准都是vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func

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