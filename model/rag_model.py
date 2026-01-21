"""
RAG 모델 모듈
멀티 쿼리 생성 및 ReRank 기능을 포함한 RAG 시스템
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings  # HuggingFace 임베딩 사용으로 주석처리
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from huggingface_hub import hf_hub_download
import torch

# .env 파일 로드
load_dotenv()


def ensure_hf_chroma_assets(
    persist_dir: str | Path,
    repo_id: str,
    repo_type: str = "dataset",
    sqlite_filename: str = "chroma.sqlite3",
    zip_filename: str = "ea99f6f7-3cf3-4ae6-8b86-dcabc4b70a9c-20260120T081052Z-1-001.zip",
    token: Optional[str] = None
) -> Path:
    """
    HF Hub에서 chroma.sqlite3 + zip 파일을 다운로드 후 persist_dir에 배치
    기존 파일이 존재하면 삭제 후 새로 다운로드
    
    Args:
        persist_dir: ChromaDB 데이터를 저장할 경로
        repo_id: 허깅페이스 리포지토리 ID
        repo_type: 리포지토리 타입 (기본값: "dataset")
        sqlite_filename: SQLite 파일명
        zip_filename: ZIP 파일명
        token: HF 토큰 (private 리포지토리인 경우 필요)
    
    Returns:
        persist_dir의 Path 객체
    """
    persist_dir = Path(persist_dir).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 실제 앱이 참조할 위치
    sqlite_dst = persist_dir / sqlite_filename
    zip_dst = persist_dir / zip_filename

    # 기존 파일이 존재하면 삭제
    if sqlite_dst.exists():
        print(f"🗑️  기존 {sqlite_filename} 삭제 중...")
        sqlite_dst.unlink()
    
    if zip_dst.exists():
        print(f"🗑️  기존 {zip_filename} 삭제 중...")
        zip_dst.unlink()

    # HF 캐시에 다운로드 (경로 반환)
    print(f"📥 {sqlite_filename} 다운로드 중...")
    sqlite_cache = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=sqlite_filename,
        token=token,
    ))
    print(f"📥 {zip_filename} 다운로드 중...")
    zip_cache = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=zip_filename,
        token=token,
    ))

    # 실제 앱이 참조할 위치로 "복사" (캐시 파일 직접 수정/이동 방지)
    shutil.copy2(sqlite_cache, sqlite_dst)
    shutil.copy2(zip_cache, zip_dst)
    
    print(f"✅ {sqlite_filename} 다운로드 완료")
    print(f"✅ {zip_filename} 다운로드 완료")

    return persist_dir


def get_device():
    """사용 가능한 디바이스 반환"""
    if torch.cuda.is_available():
        return "cuda"       # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps"        # Mac Apple Silicon 
    else:
        return "cpu"        # CPU


class QueryGeneration(BaseModel):
    """쿼리 생성 모델"""
    queries: list[str] = Field(..., description="검색 쿼리 목록")


class RAGModel:
    """멀티 쿼리 및 ReRank 기능을 포함한 RAG 모델"""
    
    def __init__(
        self,
        chromadb_path: str = "./chromaDB_data",
        collection_name: str = "med_knowledge",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "gpt-5-mini",
        temperature: float = 0,
        retrieval_k: int = 10,
        rerank_top_n: int = 3,
        rerank_model_name: str = "zeroentropy/zerank-2",
        hf_repo_id: str = "yj512/likelion_project2_chromadb",
        hf_repo_type: str = "dataset",
        sqlite_filename: str = "chroma.sqlite3",
        zip_filename: str = "ea99f6f7-3cf3-4ae6-8b86-dcabc4b70a9c-20260120T081052Z-1-001.zip",
        download_from_hf: bool = True
    ):
        """
        RAG 모델 초기화
        
        Args:
            chromadb_path: ChromaDB 데이터 경로
            collection_name: ChromaDB 컬렉션 이름
            embedding_model: Embedding 모델 이름 (HuggingFace 모델)
            llm_model: LLM 모델 이름
            temperature: LLM temperature
            retrieval_k: 검색할 문서 개수
            rerank_top_n: ReRank 후 반환할 문서 개수
            rerank_model_name: ReRank 모델 이름
            hf_repo_id: 허깅페이스 리포지토리 ID
            hf_repo_type: 허깅페이스 리포지토리 타입
            sqlite_filename: SQLite 파일명
            zip_filename: ZIP 파일명
            download_from_hf: 허깅페이스에서 다운로드 여부 (기본값: True)
        """
        # 환경 변수에서 API 키 로드
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
        
        # HF_TOKEN이 있으면 사용
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
        
        # 허깅페이스에서 ChromaDB 파일 다운로드
        if download_from_hf:
            print("🔄 허깅페이스에서 ChromaDB 파일 다운로드 중...")
            ensure_hf_chroma_assets(
                persist_dir=chromadb_path,
                repo_id=hf_repo_id,
                repo_type=hf_repo_type,
                sqlite_filename=sqlite_filename,
                zip_filename=zip_filename,
                token=self.hf_token
            )
            print("✅ ChromaDB 파일 다운로드 완료")
        
        # 디바이스 설정
        device = get_device()
        print(f"사용 중인 디바이스: {device}")
        
        # Embedding 모델 초기화 (HuggingFace 임베딩 사용)
        print(f"🔄 임베딩 모델 ({embedding_model}) 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ 임베딩 모델 로딩 완료")
        
        # # OpenAI 임베딩 모델 (주석처리)
        # self.embeddings = OpenAIEmbeddings(
        #     model=embedding_model,
        #     openai_api_key=self.openai_api_key
        # )
        
        # ChromaDB 벡터 스토어 초기화
        self.vectorstore = Chroma(
            persist_directory=chromadb_path,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Retriever 초기화
        self.retriever = self.vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': retrieval_k}
        )
        
        # LLM 모델 초기화
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=self.openai_api_key
        )
        
        # 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_template('''
다음 문맥만 고려하여 질문에 답하세요.

문맥 : """
{context}
"""
질문 : {question}
''')
        
        # 멀티 쿼리 생성 프롬프트
        self.query_generation_prompt = ChatPromptTemplate.from_template("""\
질문에 대해서 벡터 데이터베이스에서 관련 문서를 검색하기 위한
3개의 서로 다른 쿼리를 생성하세요.
거리 기반 유사성 검색의 한계를 극복하기 위해
사용자의 질문에 대해 여러 관점을 제공하는것이 목표입니다

질문 :{question}
""")
        
        # 멀티 쿼리 생성 체인
        self.query_generation_chain = (
            self.query_generation_prompt
            | self.llm.with_structured_output(QueryGeneration)
            | (lambda x: x.queries)
        )
        
        # ReRank 모델 초기화 (초기화 시점에 미리 로드)
        self.rerank_top_n = rerank_top_n
        self.rerank_model_name = rerank_model_name

        # 🔧 ReRank 모델 lazy loading (처음 rerank가 실제로 필요해질 때 로드)
        self._rerank_model = None
        
        # ReRank 체인을 초기화 시점에 미리 생성 (재사용)
        self.rerank_chain = (
            {
                "question": RunnablePassthrough(),
                "docs": self.query_generation_chain | self.retriever.map(),
            }
            | RunnableLambda(lambda x: self._rerank_topn(x, top_n=self.rerank_top_n))
        )
        
        # 멀티 쿼리 + ReRank RAG 체인 (항상 ReRank 사용)
        self.rag_chain = {
            "question": RunnablePassthrough(),
            "context": self.rerank_chain,
        } | self.prompt | self.llm | StrOutputParser()
    
    def _flatten_dedup(self, docs_nested):
        """
        중첩된 문서 리스트를 평탄화하고 중복 제거
        
        Args:
            docs_nested: list[list[Document]] 또는 list[Document]
        
        Returns:
            중복이 제거된 Document 리스트
        """
        if not docs_nested:
            return []
        
        if isinstance(docs_nested[0], Document):
            flat = docs_nested
        else:
            flat = [d for sub in docs_nested for d in sub]
        
        # 중복 제거(텍스트 기준)
        seen = set()
        uniq = []
        for d in flat:
            key = d.page_content
            if key not in seen:
                seen.add(key)
                uniq.append(d)
        return uniq

    def _get_rerank_model(self):
        """ReRank 모델을 필요 시점에 1회 로드해서 재사용"""
        if self._rerank_model is None:
            device = get_device()
            print("🔄 ReRank 모델 lazy loading...")
            # CrossEncoder는 내부적으로 HF 모델을 사용하며 device는 'cpu'/'cuda' 문자열을 받음
            # CUDA 환경이면 float16 사용을 시도하고, 아니면 기본 dtype으로 둠
            model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
            self._rerank_model = CrossEncoder(
                self.rerank_model_name,
                trust_remote_code=True,
                device=device,
                model_kwargs=model_kwargs,
            )
            print("✅ ReRank 모델 로딩 완료")
        return self._rerank_model
    
    def _rerank_topn(self, payload, top_n: Optional[int] = None):
        """
        문서를 ReRank하여 상위 N개 반환
        
        Args:
            payload: {"question": str, "docs": list[list[Document]] 또는 list[Document]}
            top_n: 반환할 문서 개수 (기본값: self.rerank_top_n)
        
        Returns:
            ReRank된 Document 리스트
        """
        if top_n is None:
            top_n = self.rerank_top_n
        
        q = payload["question"]
        docs = self._flatten_dedup(payload["docs"])
        
        if not docs:
            return []

        reranker = self._get_rerank_model()
        scored = []
        for d in docs:
            # batch size 1로 호출 (pad_token 오류 회피)
            s = float(reranker.predict([(q, d.page_content)])[0])
            scored.append((s, d))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_n]]
    
    def query(
        self,
        question: str,
        config: Optional[dict] = None
    ) -> str:
        """
        질문에 대한 답변 생성 (항상 ReRank 사용)
        
        Args:
            question: 사용자 질문
            config: LangChain 실행 설정 (선택사항)
        
        Returns:
            생성된 답변
        """
        # 질문 처리 (항상 ReRank 포함)
        if config:
            return self.rag_chain.invoke(question, config=config)
        return self.rag_chain.invoke(question)

