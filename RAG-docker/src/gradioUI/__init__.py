import os
import gradio as gr
import json
from llm import LLM


class GradioUI:
    def __init__(self, llm: LLM):
        """입력받은 llm argument를 데이터로 하는 gradioUI 객체 생성"""

        self.llm = llm
        self.answer_info = None

    def _get_unique_files(self) -> str:
        """현재 벡터디비에 있는 파일정보들을 표시"""

        unique_list = list(
            {doc["source"] for doc in self.llm.vectordb.get()["metadatas"]}
        )

        return json.dumps(unique_list, indent=4, default=str)

    def _get_sources(self):
        """chat Interface 에서 받은 LLM answer가 어떤 source 를 참고해서 나온 답변인지 확인하는 함수"""
        if not self.answer_info:
            return "if you want to see source citations, chat first."

        return json.dumps(self.answer_info, indent=4, default=str)

    def _get_model_info(self) -> str:
        """모델 정보 반환"""

        model_details = (
            f"\nGeneral Model Info:\n"
            f"\n-------------------\n"
            f"\n llm_model_name: {self.llm.llm_model_name} \n"
            f"\n Model config: {self.llm.model} \n"
            f"\nGeneral Embeddings Info:\n"
            f"\n-------------------\n"
            f"\n Embeddings model config: {self.llm.embeddings} \n"
        )

        return model_details

    def _process_input(
        self,
        question: str,
        chat_history: list[list[str]],
        rag_toggle: bool,
        system_prompt: str,
        source_docs_qty: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: float,
        repetition_penalty: float,
    ):
        """유저가 chatInterface 에서 입력한 설정 및 쿼리를 기반으로 LLM 실행 및 결과반환"""

        if rag_toggle:
            self.answer_info = self.llm.rag_mode(
                system_prompt,
                max_new_tokens,
                top_p,
                top_k,
                temperature,
                repetition_penalty,
                source_docs_qty,
            )({"query": question})
        else:
            self.answer_info = self.llm.normal_mode(
                system_prompt,
                max_new_tokens,
                top_p,
                top_k,
                temperature,
                repetition_penalty,
            )({"question": question})

        return self.llm.answer()

    def _chat_interface(self):
        """gradio 채팅 인터페이스 컴포넌트 반환"""

        return gr.ChatInterface(
            fn=self._process_input,
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
                [
                    "What are some of the results of the Dell Generative AI Pulse Survey?"
                ],
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
                [
                    "Summarize the significant developments from Dell's latest SEC filings."
                ],
            ],
            stop_btn=None,
        )

    def _ui(self):
        """gradioUI 컴포넌트를 구성하는 함수"""

        theme = gr.themes.Default()
        with gr.Blocks(
            theme=theme, css="style.css", title="Docker Concierge Chat"
        ) as ui:
            gr.Markdown(
                """
            # Retrieval Assistant
            """
            )

            with gr.Tab("Chat Session"):
                self._chat_interface().render()

            with gr.Tab("Source Citations"):
                source_text_box = gr.Textbox(label="Reference Sources")
                get_source_button = gr.Button("Get Source Content")
                get_source_button.click(
                    fn=self._get_sources, inputs=None, outputs=source_text_box
                )

            with gr.Tab("Database Files"):
                files_text_box = gr.Textbox(label="Uploaded Files")
                get_files_button = gr.Button("List Uploaded Files")
                get_files_button.click(
                    fn=self._get_unique_files, inputs=None, outputs=files_text_box
                )

            with gr.Tab("Model Info"):
                model_info_text_box = gr.Textbox(label="Model Info")
                model_info_button = gr.Button("Get Model Info")
                model_info_button.click(
                    fn=self._get_model_info, inputs=None, outputs=model_info_text_box
                )

        return ui

    def run(self):
        """gradio 웹 UI를 실행시키는 함수"""

        ui = self._ui()
        ui.queue(max_size=5)
        ui.launch(
            share=False,
            debug=True,
            server_name="0.0.0.0",
            allowed_paths=["images/dell-logo-sm.jpg"],
        )
