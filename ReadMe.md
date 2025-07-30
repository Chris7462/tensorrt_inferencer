# FCN Segmentation TensorRT backend

This is the library for model inference using the TensorRT engine with stream.

## Generate the ONNX file
This will generate the onnx file in the `models` directory.
```python
python3 script/export_fcn_to_onnx.py --height 374 --width 1238 --model fcn_resnet101 --output-dir models
```
Note: The model can be `fcn_resnet50` or `fcn_resnet101`.

## Convert to TensorRT engine
```bash
trtexec --onnx=fcn_resnet101_1238x374.onnx --saveEngine=fcn_resnet101_1238x374.engine
```
