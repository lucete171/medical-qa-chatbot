"""
RAG 모델 모듈
멀티 쿼리 생성 및 ReRank 기능을 포함한 RAG 시스템
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# .env 파일 로드
load_dotenv()


class QueryGeneration(BaseModel):
    """쿼리 생성 모델"""
    queries: list[str] = Field(..., description="검색 쿼리 목록")


class RAGModel:
    """멀티 쿼리 및 ReRank 기능을 포함한 RAG 모델"""
    
    def __init__(
        self,
        chromadb_path: str= "./chroma_data" ,
        collection_name: str = "medical_qa",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-5-mini",
        temperature: float = 0,
        retrieval_k: int = 10,
        rerank_top_n: int = 8,
        rerank_model_name: str = "zeroentropy/zerank-2"
    ):
        """
        RAG 모델 초기화
        
        Args:
            chromadb_path: ChromaDB 데이터 경로
            collection_name: ChromaDB 컬렉션 이름
            embedding_model: Embedding 모델 이름
            llm_model: LLM 모델 이름
            temperature: LLM temperature
            retrieval_k: 검색할 문서 개수
            rerank_top_n: ReRank 후 반환할 문서 개수
            rerank_model_name: ReRank 모델 이름
        """
        # 환경 변수에서 API 키 로드
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
        
        # Embedding 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=self.openai_api_key
        )
        
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
        
        # HF_TOKEN이 있으면 사용
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
        
        # ReRank 모델을 초기화 시점에 미리 로드 (한 번만 로드되고 재사용)
        print("🔄 ReRank 모델 로딩 중...")
        self.rerank_model = CrossEncoder(
            self.rerank_model_name,
            trust_remote_code=True
        )
        print("✅ ReRank 모델 로딩 완료")
        
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
        
        # ReRank 모델은 이미 초기화 시점에 로드됨
        scored = []
        for d in docs:
            # batch size 1로 호출 (pad_token 오류 회피)
            s = float(self.rerank_model.predict([(q, d.page_content)])[0])
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
            result = self.rag_chain.invoke(question, config=config)
        else:
            result = self.rag_chain.invoke(question)
        
        return result

