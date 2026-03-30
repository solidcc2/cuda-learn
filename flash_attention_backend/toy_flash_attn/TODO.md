# 优化项

- [ ] flash_attention_varlen_func支持同时从block table和连续kv中读取历史kv,这样对于新query拼接历史上下文的场景，访存更友好。可以在prefill时，始终先计算再刷新cache。

- [] do_kv_cache_update拆到单独的cuda kernel/graph完成