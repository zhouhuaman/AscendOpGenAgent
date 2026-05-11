严格按照算子生成SKILL，完成如下任务：
【任务】
参考如下torch基线代码，生成对应的Triton全融合算子decode_mla
【torch基线】
def decode_mla_golden(
    query: torch.Tensor,
    key_cache_nope: torch.Tensor,
    value_cache: torch.Tensor,
    key_cache_rope: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    q_head_num = query.shape[1]
    _, block_size, num_kv_heads, qk_nope_dim = key_cache_nope.shape
    qk_rope_dim = key_cache_rope.shape[-1]
    v_head_dim = value_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        # Split Q into Q_nope and Q_rope
        q_nope = q[:, :, :qk_nope_dim]
        q_rope = q[:, :, qk_nope_dim:]

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k_nope = key_cache_nope[block_indices].view(-1, num_kv_heads, qk_nope_dim)[
            :kv_len
        ]
        k_rope = key_cache_rope[block_indices].view(-1, num_kv_heads, qk_rope_dim)[
            :kv_len
        ]
        v = value_cache[block_indices].view(-1, num_kv_heads, v_head_dim)[:kv_len]

        if q_head_num != num_kv_heads:
            assert (
                q_head_num % num_kv_heads == 0
            ), "q_head_num must be divisible by num_kv_heads"
            rep_factor = q_head_num // num_kv_heads
            k_nope = torch.repeat_interleave(k_nope, rep_factor, dim=1)
            k_rope = torch.repeat_interleave(k_rope, rep_factor, dim=1)
            v = torch.repeat_interleave(v, rep_factor, dim=1)

        qk_nope = torch.einsum("qhd,khd->hqk", q_nope, k_nope).float()
        qk_rope = torch.einsum("qhd,khd->hqk", q_rope, k_rope).float()
        qk = (qk_nope + qk_rope) * scale
        score = torch.softmax(qk, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", score, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def test_mla(B, S, H_Q, H_KV, D_Q, D_V, dtype):
    device = get_device()
    torch.manual_seed(2)
    seq_lens = torch.full((B,), S, device=device, dtype=torch.int32)
    page_size = 128
    max_page_num = (S + page_size - 1) // page_size

    q = torch.randn((B, H_Q, D_Q), device=device, dtype=dtype)
    k_nope = torch.randn(
        (max_page_num * B, page_size, H_KV, D_V), device=device, dtype=dtype
    )
    k_rope = torch.randn(
        (max_page_num * B, page_size, H_KV, D_Q - D_V), device=device, dtype=dtype
    )
    block_table = torch.arange(
        0, B * max_page_num, device=device, dtype=torch.int32
    ).reshape(B, max_page_num)

    attn_logits = torch.empty((B, H_Q, D_V), device=device, dtype=dtype)
    sm_scale = 1.0 / (D_Q**0.5)

    decode_mla(
        q,
        k_nope,
        k_rope,
        attn_logits,
        seq_lens,
        sm_scale,
        page_size,
        block_table,
    )
    torch.npu.synchronize()

    q_len = B * [1]
    kv_len = seq_lens.cpu()
    attn_logits1 = decode_mla_golden(
        q, k_nope, k_nope, k_rope, q_len, kv_len, block_table, sm_scale
    )
    torch.npu.synchronize()

    print(
        "Max diff of mla vs golden: ", torch.max(torch.abs(attn_logits - attn_logits1))
    )
    assert torch.allclose(attn_logits, attn_logits1, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    dtypes = [torch.bfloat16, torch.float16]
    seq_lens = [4096, 3589, 1314, 128]
    
    mla_configs = [
        (16, 8, 1, 576, 512),
        (16, 32, 1, 576, 512),
        (16, 64, 1, 576, 512),
        (16, 128, 1, 576, 512),
    ]
    for dtype in dtypes:
        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in mla_configs:
                test_mla(B, S, H_Q, H_KV, D, D_V, dtype)


【精度、性能与泛化性要求】
精度参考基线验证标准。
优先生成基线case的高性能算子，但需保证生成的kernel也支持其他shape的泛化性。



【要求】
在任务进行中，实时增量记录算子生成的详细日志和调试记录到工作目录的report.md（使用中文记录）中.让我能够及时跟踪算子生成进展。
任务结束后，请输出算子生成任务的一些信息在md文件中，包括但不限于：
编译错误或精度调试记录：进行了多少轮调试，每一轮大概是什么错误。第一次生成代码后，调试了几轮才使得精度符合要求。共花费了多少token。
性能调试记录：进行了多少轮调试，每一轮调试后性能是否有提升。共花费了多少token。
任务结果：是否完成任务，精度情况怎么样，性能情况怎么样，共花费了多少时间。
【运行权限】
请所有过程都自主运行，包含给你所有目录的查询和编辑权限，不要询问我。优先选择全融合方案。需要做好report.md的记录。


