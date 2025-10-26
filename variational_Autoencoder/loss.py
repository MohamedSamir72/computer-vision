import torch
from torchvision.models import vgg16, VGG16_Weights

def get_features(x, model, layers):
    features = []
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features.append(x)
    return features

def VAE_loss(x_recon, x, mu, logvar, 
             reconstruction_loss_weight=1.0, 
             kld_loss_weight=1.0,
             perceptual_loss_act=False):
    """
    Reconstruction loss
    """
    mse = torch.mean((x_recon - x) ** 2)

    """
    KL Divergence
    """
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))


    """
    Perceptual Loss
    """
    perceptual_loss = 0

    # Load pre-trained VGG16 model for perceptual loss
    if perceptual_loss_act:
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features.eval().to(x.device)

    # Freeze VGG parameters
    for param in vgg_model.parameters():
        param.requires_grad = False

        if vgg_model is not None:
            # Get the device of the VGG model
            vgg_device = next(vgg_model.parameters()).device

            # Convert grayscale to RGB if needed (VGG expects 3 channels)
            if x.shape[1] == 1:
                x_rgb = x.repeat(1, 3, 1, 1).to(vgg_device)
                x_recon_rgb = x_recon.repeat(1, 3, 1, 1).to(vgg_device)
            else:
                x_rgb = x.to(vgg_device)
                x_recon_rgb = x_recon.to(vgg_device)

            # Resize images to at least 64x64 for VGG (28x28 is too small for deep layers)
            if x_rgb.shape[-1] < 64 or x_rgb.shape[-2] < 64:
                x_rgb = torch.nn.functional.interpolate(x_rgb, size=(64, 64), mode='bilinear', align_corners=False)
                x_recon_rgb = torch.nn.functional.interpolate(x_recon_rgb, size=(64, 64), mode='bilinear', align_corners=False)

            # Normalize to ImageNet mean/std (VGG expects normalized inputs)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(vgg_device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(vgg_device)
            x_rgb = (x_rgb - mean) / std
            x_recon_rgb = (x_recon_rgb - mean) / std

            # Define layers to extract features from (only early layers for small images)
            # For 28x28 (upscaled to 64x64), use only early layers to avoid dimension issues
            feature_layers = ['3', '8']  # relu1_2, relu2_2

            # Extract features
            with torch.no_grad():
                target_features = get_features(x_rgb, vgg_model, feature_layers)
            recon_features = get_features(x_recon_rgb, vgg_model, feature_layers)

            # Compute perceptual loss as MSE between features
            for target_feat, recon_feat in zip(target_features, recon_features):
                perceptual_loss += torch.mean((target_feat - recon_feat) ** 2)


    return (reconstruction_loss_weight * mse) + (kld_loss_weight * kl_divergence) + perceptual_loss

if __name__ == "__main__":
    # Simple test
    x = torch.randn((4, 1, 28, 28))
    recon_x = torch.randn((4, 1, 28, 28))
    mu = torch.randn((4, 4))
    logvar = torch.randn((4, 4))
    
    print("Testing VAE loss function...")
    loss = VAE_loss(recon_x, x, mu, logvar, perceptual_loss_act=True)
    print(f"VAE Loss: {loss.item()}")
