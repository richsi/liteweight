import torch

def quantize_int8(tensor: torch.Tensor):
  """
  RTN 
  Takes in a weight tensor. 
  Outputs the quantized weight tensor in INT8 and the scaling factor.
  """
  assert(tensor.dtype == torch.float16)

  q_min, q_max = -127.0, 127.0

  max_val = tensor.abs().max()
  scale = max_val / q_max

  quantized = torch.round(tensor / scale)
  quantized = torch.clamp(quantized, q_min, q_max).to(torch.int8)

  return quantized, scale
  

def dequantize_int8(tensor: torch.Tensor, scale: torch.Tensor):
  """
  Converts INT8 tensor back to FP16
  """
  assert(tensor.dtype == torch.int8)

  fp_tensor = tensor.to(torch.float32)
  fp_tensor *= scale
  fp_tensor = fp_tensor.to(torch.float16)

  return fp_tensor
