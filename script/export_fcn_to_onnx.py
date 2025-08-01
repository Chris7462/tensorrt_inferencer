#!/usr/bin/env python3
"""Script to export a pre-trained FCN model to ONNX format for TensorRT or C++ inference."""

import argparse

import torch
import torch.nn as nn
import torchvision.models.segmentation as models


class FCNWrapper(nn.Module):
    """Wrapper to extract only the main output from FCN model."""

    def __init__(self, fcn_model):
        super(FCNWrapper, self).__init__()
        self.fcn = fcn_model

    def forward(self, x):
        output = self.fcn(x)
        return output['out'] if isinstance(output, dict) else output


def export_fcn_model(output_path, input_height, input_width):
    print('Loading FCN model with ResNet101 backbone...')

    # Load pre-trained model
    base_model = models.fcn_resnet101(weights='DEFAULT')

    model = FCNWrapper(base_model)
    model.eval()

    print('Preparing dummy input...')
    dummy_input = torch.randn(1, 3, input_height, input_width)

    print('Testing model output shape...')
    with torch.no_grad():
        output = model(dummy_input)
        print(f'Output shape: {output.shape}')

    print('Exporting to ONNX...')
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=True
    )
    print(f'ONNX model saved to: {output_path}')

    # Test the exported model
    print('\nTesting ONNX model...')
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print('✓ ONNX model validation passed')
    except ImportError:
        print('⚠ ONNX package not available - skipping model validation')
        print('  Install with: pip install onnx')
    except Exception as e:
        print(f'✗ ONNX model validation failed: {e}')

    return output_path


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=374, help='The height of the input image')
    ap.add_argument('--width', type=int, default=1238, help='The width of the input image')
    ap.add_argument('--output-dir', type=str, default='onnxs',
                    help='The path to output onnx file')
    args = vars(ap.parse_args())
    # args = {'height': 374, 'width': 1238, 'output_dir': '../models'}

    height = args['height']
    width = args['width']
    output_dir = args['output_dir']

    print(f'=== Exporting fcn_resnet101 for {height}x{width} images ===')
    export_fcn_model(output_path=f'{output_dir}/fcn_resnet101_{height}x{width}.onnx',
                     input_width=width, input_height=height)

    print('ONNX export completed.')
