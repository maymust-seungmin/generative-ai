### Dell Proof of Concept RAG chatbot
### Not for production use, for educational purposes only

## THESE VARIABLES MUST APPEAR BEFORE TORCH OR CUDA IS IMPORTED
## set visible GPU devices and order of IDs to the PCI bus order
## target the L40s that is on ID 1
import sys
import torch
from gradioUI import GradioUI
from llm import LLM


def info():
    print("___________Info___________")
    print("Python version:", sys.version)
    print("pyTorch version:", torch.__version__)
    print("CUDNN version:", torch.backends.cudnn.version())
    print("Number of CUDA Devices:", torch.cuda.device_count())
    print("Current cuda device: ", torch.cuda.current_device())
    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))


def main():
    info()
    GradioUI(llm=LLM(hf_token="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")).run()


if __name__ == "__main__":
    main()
