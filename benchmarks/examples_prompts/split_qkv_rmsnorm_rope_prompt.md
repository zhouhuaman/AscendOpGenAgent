严格按照算子生成SKILL，完成如下任务：
【任务】
参考如下torch基线代码，生成对应的Triton融合算子split_qkv_rmsnorm_rope
【torch基线】
import numpy as np
import torch
import torch_npu
from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import (
    split_qkv_rmsnorm_rope,
    split_qkvgate_gemma_rmsnorm_rope,
)


def custom_rope(q, k, sin, cos, half_rope_dim):
    sin = sin.to(torch.float32).cpu().numpy()
    cos = cos.to(torch.float32).cpu().numpy()
    x1 = q[..., :half_rope_dim]
    x2 = q[..., half_rope_dim:]
    cat_x = np.concatenate((-x2, x1), axis=-1)
    mul1 = cat_x * sin
    mul2 = q * cos
    res1 = mul1 + mul2

    x1 = k[..., :half_rope_dim]
    x2 = k[..., half_rope_dim:]
    cat_x = np.concatenate((-x2, x1), axis=-1)
    mul1 = cat_x * sin
    mul2 = k * cos
    res2 = mul1 + mul2
    return res1, res2


def rms_norm(
    input,
    norm_weight,
    norm_bias,
    eps,
):
    input = input.to(torch.float32).cpu().numpy()
    norm_weight = norm_weight.to(torch.float32).cpu().numpy()
    norm_bias = norm_bias.to(torch.float32).cpu().numpy()
    reciprocal_std = 1 / np.sqrt(np.mean(input**2, axis=-1, keepdims=True) + eps)
    out = input * reciprocal_std * norm_weight + norm_bias
    return out


def test_split_qkv_rmsnorm_rope():
    q_hidden_size = 6144
    kv_hidden_size = 1024
    head_dim = 128
    bsz = 12
    eps = 1e-6
    qkv = torch.randn(bsz, q_hidden_size + kv_hidden_size * 2).to(torch.bfloat16).npu()
    q_weight = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    k_weight = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    q_bias = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    k_bias = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    sin = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    cos = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    sin = torch.from_numpy(sin).to(torch.bfloat16).npu()
    cos = torch.from_numpy(cos).to(torch.bfloat16).npu()
    # fused kernel
    q, k, v = split_qkv_rmsnorm_rope(
        qkv,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
        q_bias=q_bias,
        k_bias=k_bias,
    )

    # split
    _q, _k, _v = qkv.split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    # norm
    _q = rms_norm(_q.reshape(-1, head_dim), q_weight, q_bias, eps)
    _k = rms_norm(_k.reshape(-1, head_dim), k_weight, k_bias, eps)
    _q = _q.reshape(bsz, 1, -1, head_dim)
    _k = _k.reshape(bsz, 1, -1, head_dim)

    # rope
    cus_q, cus_k = custom_rope(_q, _k, sin, cos, half_rope_dim=64)
    cus_q = cus_q.reshape(bsz, -1)
    cus_k = cus_k.reshape(bsz, -1)

    assert (
        np.testing.assert_allclose(
            q.to(torch.float32).cpu().numpy(),
            cus_q,
            atol=5e-2,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            k.to(torch.float32).cpu().numpy(),
            cus_k,
            atol=5e-2,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            v.to(torch.float32).cpu().numpy(),
            _v.to(torch.float32).cpu().numpy(),
            rtol=5e-3,
        )
        is None
    )
【精度、性能与泛化性要求】
精度参考如上测试的精度标准。
优先生成面向基线case的高性能算子，但需保证生成的kernel也支持其他shape的泛化性。


【要求】
在任务进行中，实时增量记录算子生成的详细日志和调试记录到工作目录的report.md（使用中文记录）中.让我能够及时跟踪算子生成进展。
任务结束后，请总结算子生成任务的一些信息在report.md文件中，包括但不限于：
编译错误或精度调试记录：进行了多少轮调试，每一轮大概是什么错误。第一次生成代码后，调试了几轮才使得精度符合要求。共花费了多少token。
性能调试记录：进行了多少轮调试，每一轮调试后性能是否有提升。共花费了多少token。
任务结果：是否完成任务，精度情况怎么样，性能情况怎么样，共花费了多少时间。
【运行权限】
请所有过程都自主运行，包含给你所有目录的查询和编辑权限，不要询问我。优先选择全融合方案,不能自主回退。
