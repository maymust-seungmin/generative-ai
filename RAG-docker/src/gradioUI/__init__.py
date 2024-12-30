import gradio as gr
from llm import LLM


class GradioUI:
    def __init__(self, llm: LLM):
        """입력받은 llm argument를 데이터로 하는 gradioUI 객체 생성"""

        self.llm = llm
        self.answer_info = None

    def _get_unique_files(self) -> str:
        """현재 벡터디비에 있는 파일정보들을 표시"""

        unique_files = {doc["source"] for doc in self.llm.vectordb.get()["metadatas"]}
        
        cnt = 1
        files = []
        for file in unique_files:
            files.append(f"{cnt}: {file}")
            cnt += 1

        return "\n".join(files)

    def _get_sources(self):
        """chat Interface 에서 받은 LLM answer가 어떤 source 를 참고해서 나온 답변인지 확인하는 함수"""
        if not self.answer_info:
            return "if you want to see source citations, chat first."

        answer = (
            f"Query: {self.answer_info["query"]}\n\n"
        )

        for i, doc in enumerate(self.answer_info["source_documents"]):
            answer += "------------------------------------\n"
            answer += f"Index: {i + 1}\n"
            answer += f"Source: {doc.metadata["source"]}\n"
            answer += f"Content: {doc.page_content}\n"
            answer += "------------------------------------\n\n"

        return answer

    def _get_model_info(self) -> str:
        """모델 정보 반환"""

        model_details = (
            f"Model\n{self.llm.llm_model_name}\n\n"
            f"Model config\n{self.llm.model}\n\n"
            f"------------------------------------------------------\n\n"
            f"Embedding Model\n{self.llm.embed_model_name}\n\n"
            f"Embedding model config\n{self.llm.embeddings}\n"
        )

        return model_details

    def process_input(
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

        a = []
        for b in self.llm._streamer:
            a.append(b)
            yield "".join(a)

        # yield self.llm.answer()

    def ui(self, chat_interface):
        """gradioUI 컴포넌트를 구성하는 함수"""

        theme = gr.themes.Default()
        with gr.Blocks(theme=theme, title="Docker Concierge Chat") as ui:
            gr.Markdown(
                """
            # Retrieval Assistant
            """
            )

            with gr.Tab("Chat Session"):
                chat_interface.render()

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

            self.gradio_ui = ui

    def run(self):
        """gradio 웹 UI를 실행시키는 함수"""

        self.gradio_ui.queue(max_size=5)
        self.gradio_ui.launch(
            share=False,
            debug=True,
            server_name="0.0.0.0",
            allowed_paths=["images/dell-logo-sm.jpg"],
        )
