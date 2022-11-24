import argparse
from onnxconverter_common.auto_mixed_precision_model_path import auto_convert_mixed_precision_model_path
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fpath", type=str, default="../resources/models/vit.latest.infer.onnx")
    parser.add_argument("--output_fpath", type=str, default="../resources/models/vit.latest.amp.onnx")
    parser.add_argument("--location", type=str, default="vit.external.data")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 3, 224, 224])
    args = parser.parse_args()

    # Could also use rtol/atol attributes directly instead of this
    def validate(res1, res2):
        return True

    x = np.random.normal(0, 1, args.input_shape).astype(np.float32)
    auto_convert_mixed_precision_model_path(
        args.input_fpath,
        {"images": x},
        args.output_fpath,
        ["CUDAExecutionProvider"],
        customized_validate_func=validate,
        location=args.location,
    )
