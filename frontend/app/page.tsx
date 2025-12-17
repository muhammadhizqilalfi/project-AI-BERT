// frontend/app/page.tsx
"use client";

import React, { useState } from "react";

type SentimentResult = {
  sentiment: string;
  confidence: number; // 0–1 dari backend
  probs: number[];
};

export default function HomePage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const charLimit = 280;

  const analyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    if (!text.trim()) {
      setError("Tolong masukkan teks tweet terlebih dahulu.");
      setLoading(false);
      return;
    }

    try {
      const res = await fetch("https://arufii12-api-bert.hf.space/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult({
        sentiment: data.sentiment,
        confidence: data.confidence,
        probs: data.probs,
      });
    } catch (err: any) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const sentimentLabel = (s: string) => {
    if (!s) return "-";
    const lower = s.toLowerCase();
    if (lower.includes("pos")) return "Positif";
    if (lower.includes("neg")) return "Negatif";
    if (lower.includes("neu")) return "Netral";
    return s;
  };

  const sentimentClass = (s?: string | null) => {
    if (!s) return "badge-neutral";
    const lower = s.toLowerCase();
    if (lower.includes("pos")) return "badge-positive";
    if (lower.includes("neg")) return "badge-negative";
    if (lower.includes("neu")) return "badge-neutral";
    return "badge-neutral";
  };

  const confidencePercent =
    result?.confidence != null ? Math.min(Math.max(result.confidence * 100, 0), 100) : 0;

  return (
    <main className="page">
      <div className="card">
        {/* Header */}
        <header className="card-header">
          <div className="pill">
            <span className="pill-dot" />
            Twitter Sentiment
          </div>
          <h1 className="title">Twitter Sentiment Analyzer</h1>
          <p className="subtitle">
            Masukkan teks tweet yang ingin kamu analisis, lalu sistem akan memprediksi apakah
            sentimennya <span className="highlight positive">positif</span>,{" "}
            <span className="highlight neutral">netral</span>, atau{" "}
            <span className="highlight negative">negatif</span>.
          </p>
        </header>

        {/* Form */}
        <section className="section">
          <label className="label" htmlFor="tweet">
            Teks Tweet
          </label>
          <div className="textarea-wrapper">
            <textarea
              id="tweet"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              maxLength={charLimit}
              className="textarea"
              placeholder="Contoh: PPKM bikin hidup susah banget..."
            />
            <div className="char-counter">
              {text.length}/{charLimit}
            </div>
          </div>

          {error && <div className="error-box">Error: {error}</div>}

          <div className="form-footer">
            <p className="note">
              *Saat ini model hanya mendukung teks dalam Bahasa Indonesia.
            </p>
            <button
              onClick={analyze}
              disabled={loading || text.trim() === ""}
              className="button"
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Menganalisis...
                </>
              ) : (
                "Analyze"
              )}
            </button>
          </div>
        </section>

        {/* Result */}
        <section className="section section-result">
          <h2 className="section-title">Hasil Analisis</h2>

          <div className={`result-card ${sentimentClass(result?.sentiment)}`}>
            {!result && !loading && (
              <p className="result-placeholder">
                Belum ada analisis. Masukkan teks tweet di atas, lalu klik <b>Analyze</b>.
              </p>
            )}

            {result && (
              <>
                <div className="result-top">
                  <div>
                    <p className="result-label">Sentimen</p>
                    <p className="result-main">{sentimentLabel(result.sentiment)}</p>
                  </div>
                  <div className="result-right">
                    <p className="result-label">Confidence</p>
                    <p className="result-main">
                      {confidencePercent.toFixed(1)}
                      <span className="result-unit">%</span>
                    </p>
                  </div>
                </div>

                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${confidencePercent}%` }}
                  />
                </div>
                <p className="confidence-caption">
                  Semakin tinggi nilai confidence, semakin yakin model terhadap prediksi
                  sentimen.
                </p>

                <div className="probs-box">
                  <p className="result-label">Probabilitas Kelas</p>
                  <ul className="probs-list">
                    {result.probs.map((p, i) => (
                      <li key={i}>
                        <span>Kelas {i + 1}</span>
                        <span>{(p * 100).toFixed(2)}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </div>
        </section>

        {/* Footer kecil */}
        <footer className="footer">
          <span>UAS Mata Kuliah Artificial Intelligence</span>
          <span className="footer-model">Model: BERT-based Sentiment Classifier</span>
        </footer>
      </div>
    </main>
  );
}
  