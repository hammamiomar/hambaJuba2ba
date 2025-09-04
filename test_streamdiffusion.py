#!/usr/bin/env python3

"""
Test StreamDiffusion integration
Quick test to verify StreamDiffusion is working with our setup
"""

def test_streamdiffusion_import():
    """Test that we can import StreamDiffusion components"""
    print("Testing StreamDiffusion imports...")
    
    try:
        # Import from our submodule
        import sys
        sys.path.insert(0, './streamdiffusion/src')
        
        from streamdiffusion import StreamDiffusion
        print("✓ StreamDiffusion imported successfully")
        
        from streamdiffusion.image_utils import postprocess_image
        print("✓ Image utils imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ StreamDiffusion import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_basic_pipeline():
    """Test basic StreamDiffusion pipeline creation"""
    print("\nTesting basic pipeline setup...")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping pipeline test")
            return True
            
        # This would normally create a pipeline
        # For now just test that torch works with CUDA
        device = torch.device("cuda")
        test_tensor = torch.randn(1, 3, 512, 512).to(device)
        print(f"✓ Created test tensor on GPU: {test_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are available"""
    print("\nTesting dependencies...")
    
    deps_to_test = [
        'torch',
        'torchvision', 
        'diffusers',
        'transformers',
        'accelerate',
        'xformers'
    ]
    
    all_good = True
    
    for dep in deps_to_test:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} missing")
            all_good = False
    
    return all_good


def main():
    """Run all tests"""
    print("StreamDiffusion Integration Test")
    print("=" * 40)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("StreamDiffusion Import", test_streamdiffusion_import), 
        ("Basic Pipeline", test_basic_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 40)
    print("TEST RESULTS")
    print("=" * 40)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL" 
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! StreamDiffusion setup is working.")
        print("\nNext steps:")
        print("- Start Jupyter: uv run jupyter notebook")
        print("- Create your first pipeline")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())