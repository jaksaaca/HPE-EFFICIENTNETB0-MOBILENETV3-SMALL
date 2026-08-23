import torch
import torch.nn as nn

def run_inference(model: nn.Module, input_tensor: torch.Tensor):
    """
    Run inference using the provided model and tensor.
    Returns:
        pitch, yaw, roll (float in degrees)
    """
    # Force to use CPU since Streamlit Cloud usually runs on CPU
    device = torch.device("cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        outputs = model(input_tensor)
        
    # Output shape is expected to be [1, 3] for pitch, yaw, roll
    outputs = outputs.squeeze(0).cpu().numpy()
    
    pitch, yaw, roll = outputs[0], outputs[1], outputs[2]
    return float(pitch), float(yaw), float(roll)
