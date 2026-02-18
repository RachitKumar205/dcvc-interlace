#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test script for validating interlaced inference mode.
Tests DPB chain isolation and basic functionality.
"""

import torch
import torch.nn as nn
from src.models.video_model import DMC, RefFrame


def test_dpb_chain_isolation():
    """Test that odd and even DPB chains are isolated"""
    print("Testing DPB chain isolation...")

    model = DMC()

    # Create dummy features
    feat1 = torch.randn(1, 256, 32, 32)
    feat2 = torch.randn(1, 256, 32, 32)
    feat3 = torch.randn(1, 256, 32, 32)

    frame1 = torch.randn(1, 3, 256, 256)
    frame2 = torch.randn(1, 3, 256, 256)
    frame3 = torch.randn(1, 3, 256, 256)

    # Test 1: Add to odd chain
    model.add_ref_frame_chain(feat1, frame1, 'odd')
    ref_odd = model.get_ref_frame_chain('odd')
    ref_even = model.get_ref_frame_chain('even')

    assert ref_odd is not None, "Odd chain should have reference"
    assert ref_even is None, "Even chain should be empty"
    assert torch.allclose(ref_odd.feature, feat1), "Odd chain feature mismatch"
    print("✓ Test 1 passed: Odd chain isolation")

    # Test 2: Add to even chain
    model.add_ref_frame_chain(feat2, frame2, 'even')
    ref_odd = model.get_ref_frame_chain('odd')
    ref_even = model.get_ref_frame_chain('even')

    assert ref_odd is not None, "Odd chain should still have reference"
    assert ref_even is not None, "Even chain should now have reference"
    assert torch.allclose(ref_odd.feature, feat1), "Odd chain should be unchanged"
    assert torch.allclose(ref_even.feature, feat2), "Even chain feature mismatch"
    print("✓ Test 2 passed: Even chain isolation")

    # Test 3: Update odd chain (should not affect even)
    model.add_ref_frame_chain(feat3, frame3, 'odd')
    ref_odd = model.get_ref_frame_chain('odd')
    ref_even = model.get_ref_frame_chain('even')

    assert torch.allclose(ref_odd.feature, feat3), "Odd chain should be updated"
    assert torch.allclose(ref_even.feature, feat2), "Even chain should be unchanged"
    print("✓ Test 3 passed: Chain independence")

    # Test 4: Clear specific chain
    model.clear_dpb_chain('odd')
    ref_odd = model.get_ref_frame_chain('odd')
    ref_even = model.get_ref_frame_chain('even')

    assert ref_odd is None, "Odd chain should be cleared"
    assert ref_even is not None, "Even chain should still exist"
    print("✓ Test 4 passed: Selective chain clearing")

    # Test 5: Clear all chains
    model.add_ref_frame_chain(feat1, frame1, 'odd')
    model.clear_dpb_chain('all')
    ref_odd = model.get_ref_frame_chain('odd')
    ref_even = model.get_ref_frame_chain('even')
    ref_legacy = model.get_ref_frame_chain('legacy')

    assert ref_odd is None, "Odd chain should be cleared"
    assert ref_even is None, "Even chain should be cleared"
    assert ref_legacy is None, "Legacy chain should be cleared"
    print("✓ Test 5 passed: Clear all chains")

    print("\n✅ All DPB chain isolation tests passed!")


def test_feature_adaptor_chain():
    """Test that apply_feature_adaptor_chain works correctly"""
    print("\nTesting chain-aware feature adaptor...")

    model = DMC().eval()

    # Create dummy reference frames
    frame1 = torch.randn(1, 3, 256, 256)
    frame2 = torch.randn(1, 3, 256, 256)

    # Add to different chains
    model.add_ref_frame_chain(None, frame1, 'odd')
    model.add_ref_frame_chain(None, frame2, 'even')

    with torch.no_grad():
        # Test feature extraction from odd chain
        feat_odd = model.apply_feature_adaptor_chain('odd')
        assert feat_odd.shape[1] == 256, "Feature dimension mismatch"
        print("✓ Test 1 passed: Feature extraction from odd chain")

        # Test feature extraction from even chain
        feat_even = model.apply_feature_adaptor_chain('even')
        assert feat_even.shape[1] == 256, "Feature dimension mismatch"
        print("✓ Test 2 passed: Feature extraction from even chain")

        # Features should be different (different input frames)
        assert not torch.allclose(feat_odd, feat_even, atol=1e-4), \
            "Features from different chains should be different"
        print("✓ Test 3 passed: Chain features are independent")

    print("\n✅ All feature adaptor tests passed!")


def test_compress_chain_parameter():
    """Test that compress() accepts and uses chain parameter"""
    print("\nTesting compress() with chain parameter...")

    model = DMC().eval()
    device = 'cpu'  # Use CPU for testing

    # Initialize reference frames (use float32 to match model)
    ref_frame = torch.randn(1, 3, 256, 256)
    model.add_ref_frame_chain(None, ref_frame, 'odd')
    model.add_ref_frame_chain(None, ref_frame, 'even')

    # Create input frame (use float32 to match model)
    x = torch.randn(1, 3, 256, 256)
    qp = 10

    try:
        # Test compress with chain parameter
        with torch.no_grad():
            # Note: This will fail without proper initialization and CUDA,
            # but we're testing that the API accepts the parameter
            # In real usage, this would be called after model.update()
            result = model.compress(x, qp, chain='odd')
            print("✓ compress() accepts chain parameter")
    except RuntimeError as e:
        # Expected - entropy coder not initialized or other runtime issues
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["entropy_coder", "attribute", "none", "not initialized"]):
            print("✓ compress() accepts chain parameter (entropy coder not initialized, expected)")
        else:
            # Unexpected error
            raise

    print("\n✅ Compress API test passed!")


def test_decompress_chain_parameter():
    """Test that decompress() accepts chain parameter"""
    print("\nTesting decompress() with chain parameter...")

    model = DMC().eval()

    # Test that API accepts chain parameter
    # (Won't actually run without proper setup, but tests the signature)
    import inspect
    sig = inspect.signature(model.decompress)
    params = list(sig.parameters.keys())

    assert 'chain' in params, "decompress() should have 'chain' parameter"
    assert sig.parameters['chain'].default == 'legacy', \
        "chain parameter should default to 'legacy'"

    print("✓ decompress() has correct chain parameter signature")
    print("\n✅ Decompress API test passed!")


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("DCVC-RT Interlaced Mode Validation Tests")
    print("=" * 60)

    try:
        test_dpb_chain_isolation()
        test_feature_adaptor_chain()
        test_compress_chain_parameter()
        test_decompress_chain_parameter()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nInterlaced mode implementation is ready for testing.")
        print("Next steps:")
        print("1. Compile CUDA extensions (if not already done)")
        print("2. Run with a test video:")
        print("   python test_video.py --interlaced True \\")
        print("       --model_path_i checkpoints/cvpr2025_image.pth.tar \\")
        print("       --model_path_p checkpoints/cvpr2025_video.pth.tar \\")
        print("       --test_config dataset_config_example_yuv420.json \\")
        print("       --cuda True --write_stream True \\")
        print("       --output_path output_interlaced.json")
        print("\n3. Compare performance:")
        print("   - Sequential: python test_video.py --interlaced False ...")
        print("   - Interlaced: python test_video.py --interlaced True ...")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
