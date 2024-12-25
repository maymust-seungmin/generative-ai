from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import CSVLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.merge import MergedDataLoader

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    TextIteratorStreamer,
)
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
import torch
from pathlib import Path


class LLM:
    def __init__(
        self,
        hf_token: str,
        embed_model_name="hkunlp/instructor-large",
        llm_model_name="meta-llama/Llama-3.2-3B-Instruct",
    ):
        """
        ### LLM 관련 Task들을 모아놓은 클래스
        hf_token: 허깅페이스 토큰
        embed_model_name: 임베딩 모델 이름
        llm_model_name: llm 모델 이름
        """
        # llm_model_name samples
        # "mistralai/Mistral-7B-Instruct-v0.2"
        # "meta-llama/Llama-3.2-3B-Instruct"

        self.memory = ConversationBufferWindowMemory(
            k=5,  ## number of interactions to keep in memory
            memory_key="chat_history",
            return_messages=True,  ## formats the chat_history into HumanMessage and AImessage entity list
            input_key="query",  ### for straight retrievalQA chain
            output_key="result",  ### for straight retrievalQA chain
        )
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self.hf_token = hf_token
        self.embeddings = self._set_embeddings(embed_model_name)
        self.vectordb = self._set_vectordb()
        self.model = self._set_model(llm_model_name)
        self._tokenizer = self._set_tokenizer(llm_model_name)
        self._streamer = TextIteratorStreamer(
            self._tokenizer, timeout=10, skip_prompt=True, skip_special_tokens=True
        )

    def _set_embeddings(self, embed_model_name: str):
        """HuggingFace에서 제공하는 Instruct 임베딩 모델 객체 반환"""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        return HuggingFaceInstructEmbeddings(
            model_name=embed_model_name, model_kwargs={"device": device}
        )

    def _set_vectordb(self):
        """Instruct 임베딩 모델과 문서들을 결합하여 크로마DB 객체 초기화 및 반환"""

        current_dir = Path(__file__).resolve().parent
        csv_path = (
            current_dir.parent.parent
            / "assets"
            / "csv"
            / "dtw24-concierge-events-04-22-24-csv.csv"
        )
        pdf_path = current_dir.parent.parent / "assets" / "pdf"

        events_loader = CSVLoader(str(csv_path), encoding="windows-1252")
        pdf_dir_loader = PyPDFDirectoryLoader(str(pdf_path))
        # ppt_loader = UnstructuredPowerPointLoader("ppt-content/pan-dell-generative-ai-presentation.pptx")

        loader_all = MergedDataLoader(loaders=[events_loader, pdf_dir_loader])

        docs = loader_all.load()

        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)

        return Chroma.from_documents(
            texts, self.embeddings, persist_directory="vector-db"
        )

    def _set_model(self, llm_model_name: str):
        """양자화 적용한 transformers 모델 객체 반환"""

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = LlamaForCausalLM.from_pretrained(
            llm_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        return model

    def _set_tokenizer(self, llm_model_name: str):
        """transformers 토크나이저 반환"""

        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=self.hf_token)
        tokenizer.use_default_system_prompt = False
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return tokenizer

    def _set_llm_pipeline(
        self,
        max_new_tokens: int,
        top_p: float,
        top_k: float,
        temperature: float,
        repetition_penalty: float,
    ):
        """llm 및 llm_reg 에서 공통으로 사용되는 파이프라인 생성"""

        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self._tokenizer,
            do_sample=True,
            streamer=self._streamer,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        return HuggingFacePipeline(pipeline=text_pipeline)

    def normal_mode(
        self,
        system_prompt: str,
        max_new_tokens: int,
        top_p: float,
        top_k: float,
        temperature: float,
        repetition_penalty: float,
    ):
        """rag가 아닌 일반적인 text to text 진행"""

        template = (
            system_prompt
            + """

Question: {question}
      
Helpful Answer:"""
        )
        prompt = PromptTemplate(template=template, input_variables=["question"])

        pipeline = self._set_llm_pipeline(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        return LLMChain(llm=pipeline, prompt=prompt)

    def rag_mode(
        self,
        system_prompt: str,
        max_new_tokens: int,
        top_p: float,
        top_k: float,
        temperature: float,
        repetition_penalty: float,
        source_docs_qty: int,
    ):
        """rag 기반 text to text 진행"""

        template = (
            system_prompt
            + """

Context: {context}

Question: {question}

Helpful Answer:"""
        )
        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        pipeline = self._set_llm_pipeline(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        return RetrievalQA.from_chain_type(
            llm=pipeline,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            retriever=self.vectordb.as_retriever(
                search_type="similarity", search_kwargs={"k": source_docs_qty}
            ),
            memory=self.memory,
            verbose=True,
            return_source_documents=True,
        )

    def answer(self) -> str:
        """normal_mode 나 rag_mode 함수 실행 후, 모델 answer 반환하는 함수"""

        return "".join(self._streamer)
