import { useEffect, useRef, useState } from "react";
import "./App.css";

// In development, use Vite's built-in /api proxy so requests are same-origin (zero CORS / zero PNA blocks)
// In production, use VITE_API_URL or fallback to http://127.0.0.1:8000
const API_URL = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace(/\/+$/, "")
  : import.meta.env.DEV
    ? "/api"
    : "http://127.0.0.1:8000";

const DEFAULT_DISEASES = [
  "Bacterial Leaf Blight",
  "Bacterial Leaf Streak",
  "Bacterial Panicle Blight",
  "Blast",
  "Brown Spot",
  "Dead Heart",
  "Downy Mildew",
  "Hispa",
  "Normal",
  "Tungro",
];

function formatLabel(label = "") {
  return label.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function App() {
  const inputRef = useRef(null);

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [dragging, setDragging] = useState(false);

  // Connection status: 'checking' | 'online' | 'offline' | 'degraded'
  const [apiStatus, setApiStatus] = useState("checking");
  const [diseaseList, setDiseaseList] = useState(DEFAULT_DISEASES);

  // Check backend health on mount
  useEffect(() => {
    let isMounted = true;

    async function checkBackend() {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 3500);

        const res = await fetch(`${API_URL}/health`, {
          signal: controller.signal,
        });
        clearTimeout(timer);

        if (!isMounted) return;

        if (res.ok) {
          const data = await res.json();
          setApiStatus(data.model_loaded ? "online" : "degraded");
          setMessage(""); // Clear any prior connection error
        } else {
          setApiStatus("offline");
        }
      } catch {
        if (isMounted) {
          setApiStatus("offline");
        }
      }

      // Fetch dynamic class list
      try {
        const classesRes = await fetch(`${API_URL}/classes`);
        if (isMounted && classesRes.ok) {
          const classesData = await classesRes.json();
          if (
            Array.isArray(classesData.classes) &&
            classesData.classes.length > 0
          ) {
            setDiseaseList(classesData.classes.map(formatLabel));
          }
        }
      } catch {
        // Retain default list
      }
    }

    checkBackend();

    return () => {
      isMounted = false;
    };
  }, []);

  function handleFile(file) {
    if (!file) return;

    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    const hasValidType =
      validTypes.includes(file.type.toLowerCase()) ||
      /\.(jpe?g|png|webp)$/i.test(file.name);

    if (!hasValidType) {
      setMessage("Please upload a JPG, PNG, or WEBP image.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setMessage("Image size must be 10 MB or smaller.");
      return;
    }

    setImage(file);
    setResult(null);
    setMessage("");

    // Use FileReader for robust inline Base64 preview
    // Never fails, cannot be revoked by React StrictMode, works across all browsers
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result || "");
    };
    reader.onerror = () => {
      setMessage("Failed to read image file.");
    };
    reader.readAsDataURL(file);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  function resetImage() {
    setImage(null);
    setPreview("");
    setResult(null);
    setMessage("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  async function handleAnalyze() {
    if (!image) {
      setMessage("Please choose or drop a rice leaf image first.");
      inputRef.current?.click();
      return;
    }

    if (loading) return;

    setLoading(true);
    setMessage("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error("The backend returned an unparseable response.");
      }

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Prediction failed with status ${response.status}.`,
        );
      }

      if (
        typeof data.predicted_class !== "string" ||
        typeof data.confidence !== "number" ||
        !Array.isArray(data.top_predictions)
      ) {
        throw new Error("The backend response has an unexpected format.");
      }

      setResult(data);
      setApiStatus("online");
    } catch (err) {
      console.error("RiceGuard prediction error:", err);

      if (err instanceof TypeError && err.message.includes("fetch")) {
        setApiStatus("offline");
        setMessage(
          "Cannot connect to the RiceGuard AI backend. Please verify that FastAPI is running on port 8000.",
        );
      } else {
        setMessage(err.message || "Prediction failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
        <a href="#home" className="brand" onClick={() => setMobileMenu(false)}>
          <span className="brand-icon">✳</span>
          <span>
            RiceGuard <span className="brand-ai">AI</span>
            <small>INTELLIGENT AGRICULTURE</small>
          </span>
        </a>

        <p className="nav-label">WORKSPACE</p>

        <a
          className="nav-item active"
          href="#home"
          onClick={() => setMobileMenu(false)}
        >
          ⌂ <span>Dashboard</span>
        </a>

        <a
          className="nav-item"
          href="#analysis"
          onClick={() => setMobileMenu(false)}
        >
          ◎ <span>Crop Analysis</span>
        </a>

        <a
          className="nav-item"
          href="#diseases"
          onClick={() => setMobileMenu(false)}
        >
          ▦ <span>Disease Library</span>
        </a>

        <div className="sidebar-bottom">
          <div className="status-card">
            <span className={`status-dot ${apiStatus}`} />
            <div>
              <strong>RiceGuard AI</strong>
              <small>
                {apiStatus === "online"
                  ? "Backend connected"
                  : apiStatus === "checking"
                    ? "Checking API..."
                    : apiStatus === "degraded"
                      ? "Model unready"
                      : "Backend offline"}
              </small>
            </div>
          </div>
          <p className="sidebar-note">Smarter farming. Healthier harvests.</p>
        </div>
      </aside>

      <main className="main-content" id="home">
        <header className="topbar">
          <button
            className="menu-toggle"
            onClick={() => setMobileMenu(!mobileMenu)}
            aria-label="Toggle navigation"
          >
            ☰
          </button>

          <div className="breadcrumb">
            Workspace <span>/</span> Dashboard
          </div>

          <div className="topbar-right">
            <span className="live-pill">
              <i /> AI workspace
            </span>
            <div className="avatar">RG</div>
          </div>
        </header>

        <section className="welcome-section">
          <div className="welcome-copy">
            <span className="eyebrow">PRECISION AGRICULTURE PLATFORM</span>

            <h1>
              Protect your crops.
              <br />
              <span>Grow with confidence.</span>
            </h1>

            <p>
              Analyze rice leaf images with your trained AI model to detect
              early symptoms of fungal, bacterial, and pest diseases.
            </p>

            <a className="hero-button" href="#analysis">
              Start crop analysis <span>→</span>
            </a>
          </div>

          <div className="welcome-art" aria-hidden="true">
            <div className="art-circle">🌾</div>
            <span className="art-spark spark-one">✳</span>
            <span className="art-spark spark-two">✦</span>
          </div>
        </section>

        <section className="stats-grid" aria-label="Platform overview">
          <article className="stat-card">
            <span className="stat-icon green">⌁</span>
            <div>
              <p>Crop focus</p>
              <h2>Rice</h2>
              <small>Oryza sativa</small>
            </div>
          </article>

          <article className="stat-card">
            <span className="stat-icon yellow">◎</span>
            <div>
              <p>Supported classes</p>
              <h2>{diseaseList.length}</h2>
              <small>Pathology and healthy classes</small>
            </div>
          </article>

          <article className="stat-card">
            <span className="stat-icon purple">◇</span>
            <div>
              <p>Image analysis</p>
              <h2>{loading ? "Running" : result ? "Complete" : "Ready"}</h2>
              <small>
                {result ? "Prediction received" : "Upload an image to begin"}
              </small>
            </div>
          </article>
        </section>

        <section className="analysis-layout" id="analysis">
          <article className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">01 / IMAGE INPUT</span>
                <h2>Analyze a rice leaf</h2>
                <p>Upload a clear image of a rice leaf for inspection.</p>
              </div>
              <span className="heading-icon">↥</span>
            </div>

            <input
              id="leaf-file-input"
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
              hidden
              onChange={(event) => handleFile(event.target.files?.[0])}
            />

            {!preview ? (
              <label
                htmlFor="leaf-file-input"
                className={`dropzone ${dragging ? "dropzone-active" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  handleFile(event.dataTransfer.files?.[0]);
                }}
              >
                <span className="upload-icon">↑</span>
                <strong>Drop your leaf image here</strong>
                <span>or browse files from your device</span>
                <span className="browse-button">Choose an image</span>
                <small>JPG, PNG, WEBP · Maximum 10 MB</small>
              </label>
            ) : (
              <div
                className={`preview-area ${dragging ? "preview-dragover" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  handleFile(event.dataTransfer.files?.[0]);
                }}
              >
                <img
                  src={preview}
                  alt="Selected rice leaf"
                  onError={() =>
                    setMessage(
                      "Image preview failed to display. Please re-select the image.",
                    )
                  }
                />

                <div className="preview-info">
                  <span className="status-dot online" />
                  <span>{image?.name}</span>
                  <div className="preview-actions">
                    <button
                      type="button"
                      className="btn-replace"
                      onClick={() => inputRef.current?.click()}
                      disabled={loading}
                    >
                      Change
                    </button>
                    <button
                      type="button"
                      className="btn-remove"
                      onClick={resetImage}
                      disabled={loading}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            )}

            {message && (
              <p className="error-message" role="alert">
                {message}
              </p>
            )}

            <div className="privacy-note">
              <span>♧</span>
              <p>
                Your image is processed by your configured RiceGuard AI backend
                only when you click Analyze.
              </p>
            </div>

            <button
              type="button"
              className="analyze-button"
              disabled={loading}
              onClick={handleAnalyze}
            >
              <span>✳</span>
              {loading ? "Analyzing image..." : "Analyze crop image"}
              <span>{loading ? "…" : "→"}</span>
            </button>
          </article>

          <article className="panel results-panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">02 / AI INSIGHTS</span>
                <h2>Analysis results</h2>
                <p>Prediction from your trained MobileNetV3-Small model.</p>
              </div>

              <span
                className={`result-badge ${result ? "result-success" : ""}`}
              >
                {loading
                  ? "Analyzing"
                  : result
                    ? "Prediction received"
                    : "Awaiting analysis"}
              </span>
            </div>

            {result ? (
              <div className="prediction-results">
                <div className="result-symbol">
                  {formatLabel(result.predicted_class) === "Normal" ? "✓" : "✳"}
                </div>

                <span className="section-kicker">PREDICTED CLASS</span>

                <h3>{formatLabel(result.predicted_class)}</h3>

                <div className="confidence-card">
                  <div className="confidence-heading">
                    <span>Top prediction confidence</span>
                    <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                  </div>

                  <div
                    className="prediction-track"
                    role="progressbar"
                    aria-label="Top prediction confidence"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(result.confidence * 100)}
                  >
                    <div
                      className="prediction-fill"
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(100, result.confidence * 100),
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <h4>Top predictions</h4>

                {result.top_predictions.map((prediction) => (
                  <div className="prediction-row" key={prediction.label}>
                    <div className="prediction-label">
                      <span>{formatLabel(prediction.label)}</span>
                      <strong>
                        {(prediction.confidence * 100).toFixed(1)}%
                      </strong>
                    </div>

                    <div className="prediction-track">
                      <div
                        className="prediction-fill"
                        style={{
                          width: `${Math.max(
                            0,
                            Math.min(100, prediction.confidence * 100),
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}

                <p className="result-disclaimer">
                  AI predictions are intended for preliminary decision support.
                  For high-risk decisions, consult a qualified agricultural
                  specialist.
                </p>
              </div>
            ) : (
              <div className="empty-results">
                <div className="scan-illustration">
                  <span className="scan-leaf">♧</span>
                  <span className="scan-corner tl" />
                  <span className="scan-corner tr" />
                  <span className="scan-corner bl" />
                  <span className="scan-corner br" />
                </div>

                <h3>
                  {loading
                    ? "Analyzing your image..."
                    : "Insights start with an image"}
                </h3>

                <p>
                  {loading
                    ? "Your image is being processed by the RiceGuard AI model."
                    : "Select a rice leaf image and run an analysis to view confidence metrics and disease classifications."}
                </p>
              </div>
            )}

            <div className="result-footer">
              <span>MODEL</span>
              <strong>RiceGuard MobileNetV3-Small</strong>
              <span className="footer-separator" />
              <span>INFERENCE</span>
              <strong>
                {loading ? "Running" : result ? "Completed" : "Ready"}
              </strong>
            </div>
          </article>
        </section>

        <section className="panel disease-panel" id="diseases">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">REFERENCE LIBRARY</span>
              <h2>Supported rice classes</h2>
              <p>Diseases and healthy patterns classified by the model.</p>
            </div>

            <span className="class-count">{diseaseList.length} CLASSES</span>
          </div>

          <div className="disease-grid">
            {diseaseList.map((disease, index) => (
              <div className="disease-item" key={disease}>
                <span className="disease-number">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span>{disease}</span>
                {disease === "Normal" && (
                  <span className="healthy-tag">Healthy</span>
                )}
              </div>
            ))}
          </div>
        </section>

        <footer className="footer">
          <a href="#home" className="footer-brand">
            ✳ RiceGuard AI
          </a>
          <span>AI-assisted crop health monitoring & disease detection</span>
          <span>Built for precision agriculture.</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
