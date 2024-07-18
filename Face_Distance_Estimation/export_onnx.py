import torch
import onnx

# Check if ONNX is installed
try:
    import onnx
    print(f"ONNX version: {onnx.__version__}")
except ImportError:
    raise ImportError("Please install ONNX by running 'pip install onnx'")

# Load the MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda')  # Move the model to the GPU
midas.eval()  # Set the model to evaluation mode

# Create a dummy input tensor with the correct shape and move it to the GPU
dummy_input = torch.randn(1, 3, 256, 256).to('cuda')

# Export the model to ONNX format
torch.onnx.export(
    midas,                  # Model to be exported
    dummy_input,            # Example input tensor
    "midas_small.onnx",     # File path where the model will be saved
    export_params=True,     # Store the trained parameter weights inside the model file
    opset_version=12,       # ONNX version to export the model to
    do_constant_folding=True,  # Whether to perform constant folding for optimization
    input_names=['input'],      # Name for the model's input
    output_names=['output'],    # Name for the model's output
    dynamic_axes={
        'input': {0: 'batch_size'},    # Allow variable batch size
        'output': {0: 'batch_size'}
    }
)
