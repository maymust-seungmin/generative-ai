### Dell Proof of Concept RAG chatbot
### Not for production use, for educational purposes only

## THESE VARIABLES MUST APPEAR BEFORE TORCH OR CUDA IS IMPORTED
## set visible GPU devices and order of IDs to the PCI bus order
## target the L40s that is on ID 1
import sys
import torch
from gradioUI import GradioUI
from llm import LLM
import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()


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

    gradio = GradioUI(llm=LLM(hf_token=os.getenv("HF_TOKEN")))
    chat_interface = gr.ChatInterface(
        fn=gradio.process_input,
        chatbot=gr.Chatbot(
            bubble_full_width=False,
            avatar_images=(
                None,
                (
                    os.path.join(
                        os.path.dirname("__file__"),
                        "../assets/image/dell-logo-sm.jpg",
                    )
                ),
            ),
        ),
        examples=[
            ## events csv content
            [
                "Which booths are found in the showcase floor at Dell Technologies World 2024?"
            ],
            ["What are some common use cases for GenAI?"],
            [
                "Where is the Charting the Generative AI landscape in healthcare session going to be held?"
            ],
            [
                "Who is hosting the Understanding GenAI as a workload in a multicloud world session?"
            ],
            [
                "What enterprise Retrieval Augmented Generation solutions does Dell offer?"
            ],
            ## Powerpoint content
            ["What are some of the results of the Dell Generative AI Pulse Survey?"],
            ## pdf content, content creation, workplace productivity
            ["What is Dell's ESG policy in one sentence?"],
            [
                "Would you please write a professional email response to John explaining the benefits of Dell Powerflex."
            ],
            [
                "Create a new advertisement for Dell Technologies PowerEdge servers. Please include an interesting headline and product description."
            ],
            [
                "Create 3 engaging tweets highlighting the key advantages of using Dell Technologies solutions for Generative AI."
            ],
            [
                "What are the key steps in designing a secure and scalable on-premises solution for GenAI workloads with Dell?"
            ],
            ["Summarize the significant developments from Dell's latest SEC filings."],
        ],
        additional_inputs=[
            gr.Checkbox(
                label="Use RAG",
                value=True,
                info="Query LLM directly or query the RAG chain",
            ),
            gr.Textbox(
                label="Persona and role for system prompt:",
                lines=3,
                value="""Your name is Andie, a helpful concierge at the Dell Tech World conference held in Las Vegas.\
                                Please respond as if you were talking to someone using spoken English language.\
                                The first word of your response should never be Answer:.\
                                You are given a list of helpful information about the conference.\
                                Your goal is to use the given information to answer attendee questions.\
                                Please do not provide any additional information other than what is needed to directly answer the question.\
                                You do not need to show or refer to your sources in your responses.\
                                Please do not make up information that is not available from the given data.\
                                If you can't find the specific information from the given context, please say that you don't know.\
                                Please respond in a helpful, concise manner.\
                                """,
            ),
            gr.Slider(
                label="Number of source docs",
                minimum=1,
                maximum=10,
                step=1,
                value=3,
            ),
            gr.Slider(
                label="Max new words (tokens)",
                minimum=1,
                maximum=2048,
                step=1,
                value=1024,
            ),
            gr.Slider(
                label="Creativity (Temperature), higher is more creative, lower is less creative:",
                minimum=0.1,
                maximum=1.99,
                step=0.1,
                value=0.6,
            ),
            gr.Slider(
                label="Top probable tokens (Nucleus sampling top-p), affects creativity:",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.9,
            ),
            gr.Slider(
                label="Number of top tokens to choose from (Top-k):",
                minimum=1,
                maximum=100,
                step=1,
                value=50,
            ),
            gr.Slider(
                label="Repetition penalty:",
                minimum=1.0,
                maximum=1.99,
                step=0.05,
                value=1.2,
            ),
        ],
        stop_btn=None,
    )
    gradio.ui(chat_interface)
    gradio.run()


if __name__ == "__main__":
    main()
