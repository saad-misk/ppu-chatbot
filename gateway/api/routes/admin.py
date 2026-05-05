from fastapi import APIRouter

router = APIRouter(prefix="/admin")


@router.get("/documents")
def list_documents():
    return {"documents": [], "message": "NLP engine not connected yet"}


@router.get("/stats")
def get_stats():
    return {"total_queries": 0, "message": "Stats coming in feat/admin-routes"}


@router.post("/upload")
def upload_pdf():
    return {"message": "Upload coming in feat/admin-routes"}