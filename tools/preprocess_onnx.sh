#python -m onnxruntime.quantization.preprocess --input resources/models/deeplab.latest.onnx --output resources/models/deeplab.latest.infer.onnx
#python -m onnxruntime.quantization.preprocess --input resources/models/yolov5.latest.onnx --output resources/models/yolov5s.latest.infer.onnx
#python -m onnxruntime.quantization.preprocess --input resources/models/resnet50.latest.onnx --output resources/models/resnet50.latest.infer.onnx
python -m onnxruntime.quantization.preprocess --input resources/models/vit.latest.onnx --output resources/models/vit.latest.infer.onnx
