"""Main entry point for hambajuba2ba"""

def main():
    """Main function for the hambajuba2ba application"""
    print("Hello from hambajuba2ba!")
    
    # Test imports
    try:
        import torch
        print(f"PyTorch {torch.__version__} available")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.device_count()} devices")
        else:
            print("CUDA not available")
    except ImportError:
        print("PyTorch not installed")
    
    try:
        import diffusers
        print(f"Diffusers {diffusers.__version__} available")
    except ImportError:
        print("Diffusers not installed")


if __name__ == "__main__":
    main()