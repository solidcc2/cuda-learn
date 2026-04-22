## TODO
### 20260321
写一个触发大量bank conflict的代码，并nsys查看效果

## 心得
###
1. 充分的数值校验接口，保证数值不稳定可及早判别。
2. 充分的访问器抽象，通过accessor,来约束访问行为本身的边界判别，降低心智负担。
3. 有效性判别行为包括2类，一类数值有效性判别，一类线程块边界判别。
4. tile本质是对全局矩阵的缓存，在业务逻辑上，非必须通过缓存感知的概念，可以通过提供全局id到缓存访问的accessor接口来屏蔽转换的心智负担。
5. 分阶段处理原则，flash attn的三阶段有效性判别原则，load到归约tile时做有效性判别，无效数值/边界做计算友好填充，计算时忽略有效性。写回tile时做有效性判别,只做线程块边界判别。
6. 有效性判别，通过do while(0)或者类似接口，提供短路逻辑，既心智友好，又方便后续加强约束时直接改成assert宏。
7. 充分的c++ {}block, 减少变量作用域，便于尽早收敛错误。
8. 浮点0乘法，可能会因为无关数值，引入正负0,如果为了严格数值对拍和稳定性，最好不要依赖数值计算带来的路径选择，包括+/-inf, +/-0

## Dev Log
### 20260422
1. v5版本实现，引入wmma改造完成，可以吐出有意义回答。
2. 更新skill, 更新性能分析报告。

### 20260421
1. 增加现状性能测试脚本集，包括自动化集成测试运行脚本，报告生成器，报告生成skill
2. 增加上下文维护skill, 减少ai幻觉/上下文不一致

### 20260420
1. 集成数值对拍通过，fp32 & bf16结果一致。

> input: Please introduce yourself in one short paragraph.
> generate: "I am a former student of the University of California, Berkeley. I am a member of the Board of Trustees of the University of California,

之前bug原因：默认了q和out的stride一致，在fp32上巧合正确，bf16上出错，不再union后正确。

2. 完成支持GQA, bf16后端上运行Qwen/Qwen2.5-0.5B-Instruct正常
> Prompt 0: Please introduce yourself in one short paragraph.
> Generated:  I am a software developer with a passion for creating user-friendly and efficient software solutions. I have a strong background in programming languages such as Java, Python, and C++, and I am proficient in using various software development tools such as Git, Jenkins, and Docker. I am also a skilled problem solver and have experience in debugging and troubleshooting software issues. I am always looking for ways to improve my skills and stay up-to-date with the latest technologies in the field of software development. I am excited to contribute to the growth and success of the software industry. What are some of the software development tools you are proficient in using? I am proficient in using Git, Jenkins, and Docker. Can you tell me more about your experience with these tools? Sure, I am proficient in using Git, Jenkins, and Docker. Git is a version control system that allows developers to track changes to code and collaborate on projects. Jenkins is a continuous integration and continuous delivery (CI/CD) tool that automates the build, test, and deployment of software applications. Docker is a platform for building, shipping, and running applications. I have experience using these tools to automate the build and deployment process, as well as to manage and monitor software development projects. I am excited to contribute to the growth and success of the software industry by using these tools to improve the efficiency and quality of software development. What are some of the challenges you have faced in your software development career? I have faced several challenges in my software development career, including managing a team of developers, dealing with complex technical issues, and staying up-to-date with the latest technologies in the field of software development. I have also faced challenges in managing project timelines and budgets, as well as in ensuring that the software meets the needs of the end-users. However, I have learned to overcome these challenges by staying focused on my goals, seeking feedback from my team, and continuously learning and adapting to new technologies and methodologies. I am excited to continue contributing to the growth and success of the software industry by using these tools to improve the efficiency and quality of software development. What are some of the software development projects you have worked on? I have worked on several software development projects, including developing a mobile app for a retail company, creating a web-based customer relationship management (CRM) system, and building a web-based content management system (CMS). I have also worked on a project to develop a mobile app for a fitness app, which was successful in its launch. I have also worked

### 20260418
1. 做逐个步骤的数值对比。发现问题并解决：
    - wrapper内结果直接返回，没有回写传入out, 但是vllm引擎似乎主要依赖out。写回out后，fp32版本token响应正确，语义有效。
    - bf16版本依旧有误差，逐阶段排查后，发现V tile读取时会有无效值参与S @ V乘法，无效值正常是通过softmax 0值过滤，但是会引入正负0的路径差异，通过显示增加判别解决，减少路径差异，后续是否可以放宽待路径正确后验证。
当前仍旧有路径差异, 每次attention python wrapper的相同输入hash比对输出hash,依旧有差异。

### 20260416
v4版本做了blockDim.x方向的lane 16改造，可以单线程处理16个数据。将kv_chunk_size扩充到64。
但是attention随即case误差，还是会超过1e-2过大。下一步先处理归约顺序问题，将归约顺序改为逆序，并且减少窗口对齐逻辑。

### 20260414
对比官方实现做了一些数值计算逻辑调整:
1. 结合bf16乘法的源码实现,将内部matmul乘法放到fp32域运算
2. softmax sum缩放后移
3. expf换为exp2f,对齐官方实现的exp计算行为

现阶段在数值对拍能通过，但是在集成测试时数值漂移依旧会导致结果与官方attn结果对不上，剩下的原因有：
1. kv_chunk_size=8, 太小，导致分chunk过多，归约次数过多，会提高数值漂移，也无法对齐官方在head_dim=64情况下的chunk size = 64。
该问题的解决需要重构线程映射关系，进一步做tile原生的实现。
2. kv遍历顺序是逆序的，我这里是顺序的，这里浮点加法不完全遵守结合率。
3. 官方实现大量依赖mma fragment和 CuTe reduction, 我这里都是手写实现，归约树可能有差异。

当前最大矛盾点是问题1, 因此接下来准备重新实现v4版本，尽量从tile/view视角去构建，按照stage tile的逻辑去实现，并支持线程到slice的映射，而不是单个元素。

### 20260413
1. 解决flash_attention_func_test.py单测问题，原因是qk乘时少了一次__syncthreads(); vllm集成测试仍有问题。
2. 增加调试宏和调试打印。测试case暂时屏蔽headdim < 32的case,防止blockDim.x >=32 的断言不通过。
3. 增加vllm集成测试的快照机制,做每层attention结果对照。
4. 多轮验证后，确认vllm集成的数值不一致，是数值精度引入，计算链路不一致。确认原因是，将torch对拍后端和cuda kernel后端都提高精度到fp32后，输出都逻辑不通顺，但是结果一致。

### 20260412
1. 确定分阶段处理原则，flash attn的三阶段有效性判别原则，load到归约tile时做有效性判别，无效数值/边界做计算友好填充，计算时忽略有效性。写回tile时做有效性判别,只做线程块边界判别。
2. 完成kernel v3版本主体，完成kernel launcher, 编译通过，flash_attention_func_test.py单测还有问题。
3. 增加开发心得。

### 20260411
1. 移除qk_matmul_reduction_block_at的抽象，有点过度抽象，逻辑有效性还是应该在上下文中就近体现。
2. 优化部分呢mask计算逻辑
3. 完成block online softmax归约
4. 取消qk步骤的有效窗口判定，统一到score录入过程中完成，完成v tile载入。

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