const { useEffect, useState } = React;

const API_BASE_URL = "http://localhost:8000";

const App = () => {
  const [ragText, setRagText] = useState("");
  const [question, setQuestion] = useState("");
  const [storeStatus, setStoreStatus] = useState("");
  const [storeError, setStoreError] = useState(false);
  const [answer, setAnswer] = useState("");
  const [retrievedDocs, setRetrievedDocs] = useState([]);
  const [routeDecision, setRouteDecision] = useState("");
  const [ragDocs, setRagDocs] = useState([]);
  const [ragLoading, setRagLoading] = useState(false);

  const fetchRagDocs = async () => {
    setRagLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/rag/list`);
      if (!response.ok) {
        throw new Error("RAG 목록을 불러오지 못했습니다.");
      }
      const data = await response.json();
      setRagDocs(data.documents || []);
    } catch (error) {
      setRagDocs([]);
    } finally {
      setRagLoading(false);
    }
  };

  useEffect(() => {
    fetchRagDocs();
  }, []);

  const handleStore = async () => {
    const trimmed = ragText.trim();
    if (!trimmed) {
      setStoreStatus("지식을 입력해주세요.");
      setStoreError(true);
      return;
    }

    setStoreStatus("저장 중...");
    setStoreError(false);

    try {
      const response = await fetch(`${API_BASE_URL}/rag/store`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "저장 실패");
      }

      setStoreStatus("지식이 저장되었습니다.");
      setRagText("");
      fetchRagDocs();
    } catch (error) {
      setStoreStatus(`오류: ${error.message}`);
      setStoreError(true);
    }
  };

  const handleAsk = async () => {
    const trimmed = question.trim();
    if (!trimmed) {
      setAnswer("질문을 입력해주세요.");
      return;
    }

    setAnswer("답변 생성 중...");
    setRetrievedDocs([]);
    setRouteDecision("RAG (고정)");

    try {
      const response = await fetch(`${API_BASE_URL}/chat/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "질문 실패");
      }

      const data = await response.json();
      setAnswer(data.answer);
      setRetrievedDocs(data.retrieved_documents || []);
    } catch (error) {
      setAnswer(`오류: ${error.message}`);
    }
  };

  const handleRouteAsk = async () => {
    const trimmed = question.trim();
    if (!trimmed) {
      setAnswer("질문을 입력해주세요.");
      return;
    }

    setAnswer("라우팅 답변 생성 중...");
    setRetrievedDocs([]);
    setRouteDecision("판단 중...");

    try {
      const response = await fetch(`${API_BASE_URL}/chat/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "라우팅 질문 실패");
      }

      const data = await response.json();
      setAnswer(data.answer);
      setRetrievedDocs(data.retrieved_documents || []);
      setRouteDecision(data.route === "rag" ? "RAG" : "LLM");
    } catch (error) {
      setAnswer(`오류: ${error.message}`);
      setRouteDecision("오류");
    }
  };

  const handleDelete = async (documentId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/rag/${documentId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error("삭제 실패");
      }
      fetchRagDocs();
    } catch (error) {
      setStoreStatus(`삭제 오류: ${error.message}`);
      setStoreError(true);
    }
  };

  return (
    <main className="container">
      <header>
        <h1>RAG vs LLM 비교 학습</h1>
        <p>
          RAG 지식 저장 → 질문 → 응답 과정을 통해 RAG가 어떻게 동작하는지
          살펴보세요.
        </p>
      </header>

      <section className="card">
        <h2>① RAG 지식 입력</h2>
        <textarea
          value={ragText}
          onChange={(event) => setRagText(event.target.value)}
          placeholder="RAG 지식으로 저장할 문장을 입력하세요."
          rows="4"
        ></textarea>
        <button type="button" onClick={handleStore}>
          지식 저장
        </button>
        <p className="status" style={{ color: storeError ? "#dc2626" : "#16a34a" }}>
          {storeStatus}
        </p>
      </section>

      <section className="card">
        <h2>② 질문 입력</h2>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="질문을 입력하세요."
          rows="3"
        ></textarea>
        <div className="button-row">
          <button type="button" onClick={handleAsk}>
            질문하기 (RAG)
          </button>
          <button type="button" onClick={handleRouteAsk}>
            라우팅 질문하기
          </button>
        </div>
      </section>

      <section className="card">
        <h2>응답 결과</h2>
        {routeDecision && <p className="route-tag">라우팅 결과: {routeDecision}</p>}
        <div className="answer">{answer}</div>
        <h3>검색된 문서</h3>
        <ul className="retrieved">
          {retrievedDocs.map((doc, index) => (
            <li key={`${doc.text}-${index}`}>
              {doc.text} (score: {doc.score?.toFixed(3) ?? "N/A"})
            </li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h2>저장된 RAG 지식</h2>
        {ragLoading ? (
          <p>불러오는 중...</p>
        ) : ragDocs.length === 0 ? (
          <p>저장된 문서가 없습니다.</p>
        ) : (
          <ul className="retrieved">
            {ragDocs.map((doc) => (
              <li key={doc.id} className="rag-item">
                <span>
                  {doc.text}
                  {doc.created_at ? ` (${new Date(doc.created_at).toLocaleString()})` : ""}
                </span>
                <button
                  type="button"
                  className="danger"
                  onClick={() => handleDelete(doc.id)}
                >
                  삭제
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
