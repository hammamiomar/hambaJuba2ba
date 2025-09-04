"""
Modern TensorRT installation module for hambajuba2ba
Adapted from StreamDiffusion's install-tensorrt.py to work with uv
"""

import subprocess
import sys
import platform
from typing import Literal, Optional
from packaging.version import Version


def run_uv_command(command: str) -> None:
    """Run a uv command and handle errors."""
    print(f"Running: uv {command}")
    result = subprocess.run(
        f"uv {command}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running uv {command}:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError(f"Failed to run uv {command}")
    else:
        print(f"Success: {result.stdout}")


def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed via uv."""
    try:
        result = subprocess.run(
            f"uv pip show {package_name}",
            shell=True,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def get_package_version(package_name: str) -> Optional[Version]:
    """Get the version of an installed package."""
    try:
        result = subprocess.run(
            f"uv pip show {package_name}",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version_str = line.split(':', 1)[1].strip()
                    return Version(version_str)
        return None
    except Exception:
        return None


def get_cuda_version_from_torch() -> Optional[Literal["11", "12"]]:
    """Detect CUDA version from PyTorch installation."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                major_version = cuda_version.split(".")[0]
                if major_version in ["11", "12"]:
                    return major_version
        return None
    except ImportError:
        print("PyTorch not found. Please install PyTorch first.")
        return None


def install_tensorrt(cuda_version: Optional[Literal["11", "12"]] = None) -> None:
    """Install TensorRT and related dependencies using uv."""
    
    # Auto-detect CUDA version if not provided
    if cuda_version is None:
        cuda_version = get_cuda_version_from_torch()
    
    if cuda_version is None or cuda_version not in ["11", "12"]:
        print("Could not detect CUDA version or unsupported version.")
        print("Please ensure PyTorch with CUDA support is installed.")
        print("Supported CUDA versions: 11, 12")
        return
    
    print(f"Installing TensorRT for CUDA {cuda_version}...")
    
    # Check and handle existing TensorRT installation
    if is_package_installed("tensorrt"):
        current_version = get_package_version("tensorrt")
        if current_version and current_version < Version("9.0.0"):
            print("Removing old TensorRT version...")
            run_uv_command("pip uninstall -y tensorrt")
    
    # Install cuDNN
    cudnn_package = f"nvidia-cudnn-cu{cuda_version}==8.9.4.25"
    print(f"Installing {cudnn_package}...")
    run_uv_command(f"pip install {cudnn_package} --no-cache-dir")
    
    # Install TensorRT
    if not is_package_installed("tensorrt"):
        print("Installing TensorRT...")
        run_uv_command(
            "pip install --pre --extra-index-url https://pypi.nvidia.com "
            "tensorrt==9.0.1.post11.dev4 --no-cache-dir"
        )
    
    # Install polygraphy
    if not is_package_installed("polygraphy"):
        print("Installing polygraphy...")
        run_uv_command(
            "pip install polygraphy==0.47.1 "
            "--extra-index-url https://pypi.ngc.nvidia.com"
        )
    
    # Install onnx-graphsurgeon
    if not is_package_installed("onnx_graphsurgeon"):
        print("Installing onnx-graphsurgeon...")
        run_uv_command(
            "pip install onnx-graphsurgeon==0.3.26 "
            "--extra-index-url https://pypi.ngc.nvidia.com"
        )
    
    # Install pywin32 on Windows
    if platform.system() == 'Windows' and not is_package_installed("pywin32"):
        print("Installing pywin32 (Windows)...")
        run_uv_command("pip install pywin32")
    
    print("TensorRT installation complete!")
    
    # Verify installation
    print("\nVerifying installation...")
    verify_tensorrt_installation()


def verify_tensorrt_installation() -> None:
    """Verify that TensorRT is properly installed."""
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
        
        # Test basic TensorRT functionality
        logger = trt.Logger(trt.Logger.WARNING)
        print("TensorRT logger created successfully")
        
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"Available GPUs: {torch.cuda.device_count()}")
        
        print("TensorRT verification passed!")
        
    except ImportError as e:
        print(f"TensorRT import failed: {e}")
        print("Installation may have failed or TensorRT is not properly configured.")
    except Exception as e:
        print(f"TensorRT verification failed: {e}")


def main():
    """Main entry point for the installation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Install TensorRT and dependencies using uv"
    )
    parser.add_argument(
        "--cuda", 
        choices=["11", "12"], 
        help="CUDA version (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    try:
        install_tensorrt(args.cuda)
    except Exception as e:
        print(f"Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()