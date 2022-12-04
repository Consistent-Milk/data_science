import torch

printer = torch.cuda.is_available()
print(printer)

printer = torch.cuda.get_device_name()
print(printer)
