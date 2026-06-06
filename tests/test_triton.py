import torch, triton, triton.language as tl

print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))  # want (12, 0) for sm_120

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 100_000
x = torch.rand(n, device="cuda")
y = torch.rand(n, device="cuda")
out = torch.empty_like(x)
grid = (triton.cdiv(n, 1024),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
torch.cuda.synchronize()

assert torch.allclose(out, x + y), "MISMATCH — toolchain problem"
print("Triton works.")