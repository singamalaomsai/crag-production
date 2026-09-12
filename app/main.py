from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.graph.workflow import CRAGPipeline
from app.retrieval.chunking import MetadataChunker

HTML_CHAT_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CRAG Assistant</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; height: 100vh; overflow: hidden; }
    #sidebar { width: 330px; background: #1e293b; border-right: 1px solid #334155; padding: 20px; display: flex; flex-direction: column; gap: 14px; flex-shrink: 0; overflow-y: auto; }
    #chat-container { flex: 1; display: flex; flex-direction: column; height: 100vh; min-width: 0; }
    #header { padding: 16px 24px; background: #1e293b; border-bottom: 1px solid #334155; font-weight: 600; font-size: 1.1rem; }
    #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
    .message { max-width: 80%; padding: 12px 16px; border-radius: 10px; line-height: 1.5; font-size: 0.95rem; }
    .user-msg { align-self: flex-end; background: #2563eb; color: #fff; }
    .bot-msg { align-self: flex-start; background: #1e293b; border: 1px solid #334155; color: #f1f5f9; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; margin-bottom: 8px; }
    .badge-local { background: #065f46; color: #6ee7b7; }
    .badge-web { background: #7c2d12; color: #fdba74; }
    .sources { margin-top: 10px; padding-top: 8px; border-top: 1px solid #334155; font-size: 0.8rem; color: #94a3b8; }
    .sources ul { padding-left: 18px; margin-top: 4px; }
    .sources a { color: #60a5fa; text-decoration: none; word-break: break-all; }
    #input-box { padding: 16px 20px; background: #1e293b; border-top: 1px solid #334155; display: flex; gap: 10px; align-items: center; }
    #input-box input { flex: 1; min-width: 0; padding: 10px 14px; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #fff; font-size: 0.95rem; outline: none; }
    #input-box button { background: #2563eb; color: #fff; border: none; padding: 10px 22px; border-radius: 8px; font-weight: 600; cursor: pointer; flex-shrink: 0; }
    #sidebar input, #sidebar textarea { width: 100%; padding: 10px; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #fff; font-size: 0.9rem; outline: none; }
    #sidebar button { width: 100%; background: #2563eb; color: #fff; border: none; padding: 10px; border-radius: 8px; font-weight: 600; cursor: pointer; }
    input[type="file"] { font-size: 0.8rem; color: #94a3b8; background: transparent !important; border: 1px dashed #334155 !important; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .divider { border-top: 1px solid #334155; margin: 6px 0; }
  </style>
</head>
<body>
  <div id="sidebar">
    <strong style="color: #94a3b8; text-transform: uppercase; font-size: 0.85rem;">Upload PDF Document</strong>
    <p style="font-size: 0.82rem; color: #94a3b8;">Select a PDF from your device to index it automatically:</p>
    <input type="file" id="pdfFileInput" accept=".pdf">
    <button onclick="uploadPdf()" id="pdfBtn">Index PDF Document</button>

    <div class="divider"></div>

    <strong style="color: #94a3b8; text-transform: uppercase; font-size: 0.85rem;">Ingest Raw Text</strong>
    <input type="text" id="docIdInput" placeholder="Document ID (e.g., hr_policy)">
    <textarea id="docTextInput" rows="4" placeholder="Or paste plain text here..."></textarea>
    <button onclick="ingestDoc()" id="ingestBtn">Upload Text</button>

    <div id="ingestStatus" style="font-size: 0.85rem; line-height: 1.4;"></div>
  </div>

  <div id="chat-container">
    <div id="header">Corrective RAG (CRAG) Assistant</div>
    <div id="messages">
      <div class="message bot-msg">
        Upload a PDF using the left sidebar, or ask questions directly against indexed knowledge.
      </div>
    </div>
    <div id="input-box">
      <input type="text" id="questionInput" placeholder="Ask a question..." onkeydown="if(event.key==='Enter') sendQuery()">
      <button onclick="sendQuery()" id="sendBtn">Send</button>
    </div>
  </div>

  <script>
    pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

    async function extractTextFromPdf(file) {
      var arrayBuffer = await file.arrayBuffer();
      var pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
      var fullText = "";
      for (var i = 1; i <= pdf.numPages; i++) {
        var page = await pdf.getPage(i);
        var content = await page.getTextContent();
        var pageText = content.items.map(function(item) { return item.str; }).join(" ");
        fullText += pageText + "\\n\\n";
      }
      return fullText;
    }

    async function uploadPdf() {
      var fileInput = document.getElementById("pdfFileInput");
      var status = document.getElementById("ingestStatus");
      if (!fileInput.files.length) {
        status.innerHTML = '<span style="color: #f87171;">Select a PDF file first.</span>';
        return;
      }
      var file = fileInput.files[0];
      var btn = document.getElementById("pdfBtn");
      btn.disabled = true;
      status.innerHTML = '<span style="color: #60a5fa;">Extracting text from ' + file.name + '...</span>';

      try {
        var text = await extractTextFromPdf(file);
        if (!text.trim()) {
          throw new Error("No readable text found in PDF (it might be a scanned image).");
        }
        status.innerHTML = '<span style="color: #60a5fa;">Chunking and indexing into Qdrant...</span>';

        var res = await fetch("/api/v1/documents/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ doc_id: file.name, text: text })
        });
        var data = await res.json();
        status.innerHTML = '<span style="color: #4ade80;">Indexed ' + file.name + ' (' + data.chunks_created + ' chunks created)!</span>';
        fileInput.value = "";
      } catch (err) {
        status.innerHTML = '<span style="color: #f87171;">Error: ' + err.message + '</span>';
      } finally {
        btn.disabled = false;
      }
    }

    async function sendQuery() {
      var input = document.getElementById("questionInput");
      var q = input.value.trim();
      if (!q) return;

      var msgs = document.getElementById("messages");
      var userDiv = document.createElement("div");
      userDiv.className = "message user-msg";
      userDiv.textContent = q;
      msgs.appendChild(userDiv);
      input.value = "";

      var botDiv = document.createElement("div");
      botDiv.className = "message bot-msg";
      botDiv.innerHTML = "<em>Analyzing and verifying sources...</em>";
      msgs.appendChild(botDiv);
      msgs.scrollTop = msgs.scrollHeight;

      var sendBtn = document.getElementById("sendBtn");
      sendBtn.disabled = true;

      try {
        var res = await fetch("/api/v1/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q })
        });
        var data = await res.json();

        var badge = data.web_search_fallback
          ? '<div class="badge badge-web">Web Search Fallback Triggered</div>'
          : '<div class="badge badge-local">Local Document Verified</div>';

        var sourceHtml = "";
        if (data.sources && data.sources.length > 0) {
          var items = "";
          for (var i = 0; i < data.sources.length; i++) {
            var s = data.sources[i];
            if (s.doc_id.indexOf("http") === 0) {
              var title = (s.metadata && s.metadata.title) ? s.metadata.title : s.doc_id;
              items += '<li><a href="' + s.doc_id + '" target="_blank">' + title + '</a></li>';
            } else {
              items += '<li><strong>Doc ID:</strong> ' + s.doc_id + '</li>';
            }
          }
          sourceHtml = '<div class="sources"><strong>Sources:</strong><ul>' + items + '</ul></div>';
        }

        var genText = (data.generation || "").replace(/\\n/g, "<br>");
        botDiv.innerHTML = badge + '<div>' + genText + '</div>' + sourceHtml;
      } catch (err) {
        botDiv.innerHTML = '<span style="color: #f87171;">Error: ' + err.message + '</span>';
      } finally {
        sendBtn.disabled = false;
        msgs.scrollTop = msgs.scrollHeight;
      }
    }

    async function ingestDoc() {
      var docId = document.getElementById("docIdInput").value.trim();
      var text = document.getElementById("docTextInput").value.trim();
      var status = document.getElementById("ingestStatus");
      if (!docId || !text) {
        status.innerHTML = '<span style="color: #f87171;">Please enter ID and text.</span>';
        return;
      }

      var btn = document.getElementById("ingestBtn");
      btn.disabled = true;
      status.textContent = "Indexing document...";

      try {
        var res = await fetch("/api/v1/documents/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ doc_id: docId, text: text })
        });
        var data = await res.json();
        status.innerHTML = '<span style="color: #4ade80;">Indexed ' + data.chunks_created + ' chunk(s).</span>';
        document.getElementById("docIdInput").value = "";
        document.getElementById("docTextInput").value = "";
      } catch (err) {
        status.innerHTML = '<span style="color: #f87171;">Upload error: ' + err.message + '</span>';
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunker = MetadataChunker(chunk_size=128, chunk_overlap=20)
    initial_docs = [
        ("doc_1", "PostgreSQL is an open-source object-relational database system emphasizing extensibility and SQL compliance."),
        ("doc_2", "FastAPI is a modern web framework for Python with automatic interactive OpenAPI documentation."),
        ("doc_3", "Error ERR_CONN_REFUSED_502 indicates a bad gateway or unreachable upstream microservice behind a reverse proxy."),
        ("doc_4", "Qdrant is an open-source vector similarity search engine and vector database written in Rust."),
    ]

    all_chunks = []
    for doc_id, text in initial_docs:
        all_chunks.extend(chunker.split_text(text=text, doc_id=doc_id))

    pipeline = CRAGPipeline(chunks=all_chunks)
    app.state.chunker = chunker
    app.state.crag_pipeline = pipeline
    yield


app = FastAPI(
    title="Corrective RAG (CRAG) Production Service",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def serve_chat_ui():
    return HTMLResponse(content=HTML_CHAT_UI)