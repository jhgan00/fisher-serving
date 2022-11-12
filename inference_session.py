"""onnxruntime session"""

import onnxruntime as ort
import numpy as np


class BatchInferenceSession(ort.InferenceSession):

    def batch_run(self, x: np.ndarray, batch_size: int):
        """
        - 주어진 입력을 최대 batch_size 만큼씩 나누어서 처리
        - onnx의 dynamic axis 를 활용하는 경우 추론 시간이 불안정한 문제가 있어서 이렇게 사용
        """
        result = []
        io_binding = self.io_binding()
        for i in range(0, len(x), batch_size):
            X = x[i:i+batch_size]
            device_id = int(self.get_provider_options()['CUDAExecutionProvider']['device_id'])
            X_ortvalue = ort.OrtValue.ortvalue_from_numpy(X, 'cuda', device_id)
            io_binding.bind_input(
                name=self.get_inputs()[0].name,
                device_type=X_ortvalue.device_name(),
                element_type=np.float32,
                device_id=device_id,
                shape=X_ortvalue.shape(),
                buffer_ptr=X_ortvalue.data_ptr()
            )
            io_binding.bind_output(self.get_outputs()[0].name)
            self.run_with_iobinding(io_binding)
            result.append(io_binding.copy_outputs_to_cpu()[0])
        result = np.concatenate(result)
        return result


def initialize_session(model_path: str, device_id: int = 0, warmup_runs: int = 10, warmup_batch_size: int = 8)\
        -> BatchInferenceSession:
    """onnx 파일 경로와 장치 ID를 입력받아 세션을 초기화. 더미 데이터로 웜업 수행"""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = BatchInferenceSession(
        model_path,
        providers=['CUDAExecutionProvider'],
        provider_options=[{
            'device_id': device_id,
            # "cudnn_conv_use_max_workspace": '1'
        }],
        sess_options=so
    )
    inp = session.get_inputs()[0]
    inp_shape = inp.shape
    batch_size = inp_shape[0]
    if isinstance(batch_size, str):
        inp_shape[0] = warmup_batch_size

    for _ in range(warmup_runs):
        x = np.random.normal(0, 1, inp_shape).astype(np.float32)
        session.run(None, {inp.name: x})

    return session
