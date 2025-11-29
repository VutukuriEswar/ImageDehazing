import torch
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path

# Import the model from our model.py file
from model import UNet

def dehaze_image(model, image_path, device, image_size):
    """Loads an image, dehazes it, and returns the output PIL Image."""
    # Preprocessing: resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocessing: convert tensor back to PIL Image
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
    
    return output_image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load the model architecture
    model = UNet(in_channels=3, out_channels=3).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")

    if args.input_path:
        # Process single image
        output_image = dehaze_image(model, args.input_path, device, args.image_size)
        output_image.save(args.output_path)
        print(f"Dehazed image saved to {args.output_path}")
    
    elif args.input_dir:
        # Process all images in directory
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process.")
        for image_file in image_files:
            output_path = output_dir / image_file.name
            output_image = dehaze_image(model, image_file, device, args.image_size)
            output_image.save(output_path)
            print(f"Dehazed {image_file.name} and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Dehazing Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--input_path', type=str, help='Path to a single input image')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output_path', type=str, help='Path to save the single output image')
    parser.add_argument('--output_dir', type=str, help='Directory to save output images')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference if available')
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images before processing')
    
    args = parser.parse_args()
    
    if not args.input_path and not args.input_dir:
        parser.error("Either --input_path or --input_dir must be specified")
    
    if args.input_path and not args.output_path:
        parser.error("--output_path must be specified when using --input_path")
    
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir must be specified when using --input_dir")
        
    main(args)