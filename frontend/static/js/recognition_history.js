const RECOGNITION_HISTORY_KEY = "sentinel.recognition.history";
const RECOGNITION_HISTORY_LIMIT = 10;

function loadRecognitionHistory() {
  try {
    const raw = localStorage.getItem(RECOGNITION_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (_error) {
    return [];
  }
}

function saveRecognitionHistory(entries) {
  localStorage.setItem(RECOGNITION_HISTORY_KEY, JSON.stringify(entries.slice(0, RECOGNITION_HISTORY_LIMIT)));
}

function pushRecognitionHistory(entry) {
  const history = loadRecognitionHistory();
  history.unshift(entry);
  saveRecognitionHistory(history);
  return history;
}

function recognitionHistoryPage() {
  return {
    history: loadRecognitionHistory(),
    selectedId: null,

    refreshHistory() {
      this.history = loadRecognitionHistory();
    },

    async replay(entryId) {
      const entry = this.history.find((item) => item.id === entryId);
      if (!entry) return;

      this.selectedId = entryId;
      const response = await fetch("/partials/recognition/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(entry.payload),
      });
      if (!response.ok) return;

      const html = await response.text();
      const target = document.getElementById("recognition-results");
      if (target) {
        target.innerHTML = html;
        if (window.initRecognitionPanels) {
          window.initRecognitionPanels(target);
        }
      }
    },

    formatWhen(timestamp) {
      return new Date(timestamp).toLocaleString();
    },
  };
}

document.body.addEventListener("recognition:completed", (event) => {
  const detail = event.detail;
  if (!detail?.payload) return;
  pushRecognitionHistory({
    id: detail.id,
    timestamp: detail.timestamp,
    profile: detail.profile,
    face_count: detail.face_count,
    total_ms: detail.total_ms,
    payload: detail.payload,
  });
  window.dispatchEvent(new CustomEvent("recognition:history-updated"));
});

window.recognitionHistoryPage = recognitionHistoryPage;
window.loadRecognitionHistory = loadRecognitionHistory;
