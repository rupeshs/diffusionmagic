import torch


class Computing:
    __instance = None
    __fetch = True

    def __new__(cls):
        if Computing.__instance is None:
            Computing.__instance = super().__new__(cls)
        return Computing.__instance

    def __init__(self):
        if Computing.__fetch:
            self._device = self._detect_processor()
            self._torch_datatype = (
                torch.float32 if self._device == "cpu" else torch.float16
            )
            Computing.__fetch = False

    @property
    def name(self):
        """Returns the device name CPU/CUDA/MPS"""
        return self._device

    @property
    def datatype(self):
        """Returns the optimal data type for interference"""
        return self._torch_datatype

    def _detect_processor(self) -> str:
        if torch.cuda.is_available():
            current_device_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device_index)
            print(f"DEVICE: {gpu_name} ")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple silicon (M1/M2) hardware
            print("DEVICE: MPS backend")
            return "mps"
        else:
            print("DEVICE: GPU not found,using CPU")
            return "cpu"
