from model.rag_model import RAGModel

_rag_model = None

def load_model() -> RAGModel:
    global _rag_model
    if _rag_model is None:
        _rag_model = RAGModel()
    return _rag_model
