import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pth_file = "prune/resnet50.pth"
onnx_file = "prune/resnet50.onnx"
model = torch.load(pth_file, map_location=device)
dummy_input = torch.randn(1, 3, 32, 32, device=device)
torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    opset_version=10,
    verbose=True,
    input_names=["input"],
    output_names=["output"]
)