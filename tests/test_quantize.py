import torch
from liteweight import quantize as Q

def test_fakequant_matmul(tensor: torch.Tensor):

  q_tensor, scale = Q.quantize_int8(tensor)
  dq_tensor = Q.dequantize_int8(q_tensor, scale)

  error = torch.abs(tensor - dq_tensor)

  flat_error = error.flatten()
  max_idx = flat_error.argmax()

  val_1 = tensor.flatten()[max_idx]
  val_2 = dq_tensor.flatten()[max_idx]

  print(f"Original tensor:\n{tensor}")
  print(f"Quantized: \n{q_tensor}\nScale: {scale}")
  print(f"Dequantized: \n{dq_tensor}")
  print(f"Error: \n{error}")
  print(f"Largest Error: {error.max()}")
  print(f"Came from: {val_1.item():.4f} (Tensor 1) and {val_2.item():.4f} (Tensor 2)")
  assert torch.allclose(tensor, dq_tensor, atol=0.01)


if __name__ == "__main__":
  torch.manual_seed(42)
  tensor = torch.randn(4, 4, dtype=torch.float16)
  test_fakequant_matmul(tensor)