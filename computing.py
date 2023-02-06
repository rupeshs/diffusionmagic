import torch


class Computing():
    __instance = None
    __fetch = True
    def __new__(cls):
        print(cls.__instance)
        if Computing.__instance is None:
            Computing.__instance = super().__new__(cls)
        return Computing.__instance
    
    def __init__(self):
        self._device = None
        self._torch_datatype = None
        if Computing.__fetch:
            self._device=self._detect_processor()
            self._torch_datatype = torch.float32 if self._device=="cpu" else torch.float16
            Computing.__fetch=False

    @property
    def name(self):
        return self._device

    @property
    def datatype(self):
        return self._torch_datatype

    def _detect_processor(self)->str:
        if torch.cuda.is_available():
            current_device_index = torch.cuda.current_device()
            gpu_name =  torch.cuda.get_device_name(current_device_index)
            print(f"DEVICE: {gpu_name} detected")
            return "cuda"
        else:
            print("DEVICE: GPU not found,using CPU" )
            return "cpu"