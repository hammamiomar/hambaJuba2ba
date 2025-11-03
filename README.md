# hambajuba2ba

Real-time diffusion pipeline with
## Setup

1. uv venv --python /usr/bin/python3.11

2. source .venv/bin/activate

3. uv pip install -e '.[all]'


4. Verify CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

 import tensorrt as trt
    print(f"\nTensorRT version: {trt.__version__}")


5. benchmark
packages/streamdiffusion/examples/benchmark/single.py",
        "--acceleration", "tensorrt",
        "--iterations", "10",
        "--warmup", "3"
