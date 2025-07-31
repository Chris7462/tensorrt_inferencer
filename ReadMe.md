# FCN Segmentation TensorRT backend

This is the library for model inference using the TensorRT engine with stream.

## Generate the ONNX file
This will generate the onnx file in the `models` directory.
```python
python3 script/export_fcn_to_onnx.py --height 374 --width 1238 --output-dir models
```
Note: The model can be `fcn_resnet50` or `fcn_resnet101`.

## Convert to TensorRT engine
```bash
trtexec --onnx=./models/fcn_resnet101_374x1238.onnx \
        --saveEngine=./engines/fcn_resnet101_374x1238.engine \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --verbose
```
