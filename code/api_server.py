"""
RAG系统 FastAPI 后端服务（现代 lifespan 写法）
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager  # 新增：生命周期管理器
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入原有RAG系统
from RecipeRAGSystem import RecipeRAGSystem, DEFAULT_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局单例 RAG系统
rag_system = None


# ===================== 现代 Lifespan 写法 =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：启动时初始化RAG，关闭时可做清理
    """
    global rag_system
    # -------------- 启动时执行（替代 startup）--------------
    try:
        rag_system = RecipeRAGSystem(DEFAULT_CONFIG)
        rag_system.initialize_system()
        rag_system.build_knowledge_base()
        logger.info("✅ RAG系统初始化完成，知识库加载成功！")
    except Exception as e:
        logger.error(f"❌ RAG系统初始化失败：{str(e)}")
        raise e

    # 挂载应用运行
    yield

    # -------------- 关闭时执行（替代 shutdown，可选）--------------
    logger.info("🔌 服务已关闭，RAG系统已退出")


# 初始化FastAPI，传入 lifespan
app = FastAPI(
    title="食谱RAG智能问答系统", version="1.0", lifespan=lifespan  # 关键：绑定生命周期
)

# ===================== 跨域配置（不变）=====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求体模型（不变）
class QuestionRequest(BaseModel):
    question: str
    stream: bool = True


# ===================== API接口（完全不变）=====================
@app.get("/", summary="访问前端页面")
def get_frontend():
    return FileResponse("index.html")


@app.post("/api/ask", summary="提问接口")
async def ask_question(request: QuestionRequest):
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG系统未初始化完成")

    try:
        if request.stream:

            def generate():
                for chunk in rag_system.ask_question(request.question, stream=True):
                    yield chunk

            return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
        else:
            answer = rag_system.ask_question(request.question, stream=False)
            return {"answer": answer}

    except Exception as e:
        logger.error(f"提问失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败：{str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
