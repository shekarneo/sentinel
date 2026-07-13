const API_V1 = "/api/v1";

document.addEventListener("alpine:init", () => {
  Alpine.store("console", {
    loading: false,
    healthStatus: "checking",
    toasts: [],
    toastId: 0,

    async checkHealth() {
      try {
        const response = await fetch(`${API_V1}/system/health`);
        const data = await response.json();
        this.healthStatus = data.status || "unknown";
      } catch (_error) {
        this.healthStatus = "error";
      }
    },

    setLoading(value) {
      this.loading = value;
    },

    showToast(title, message) {
      const id = ++this.toastId;
      this.toasts.push({ id, title, message });
      setTimeout(() => this.removeToast(id), 4500);
    },

    removeToast(id) {
      this.toasts = this.toasts.filter((toast) => toast.id !== id);
    },
  });
});

function consoleApp() {
  return {
    get loading() {
      return Alpine.store("console").loading;
    },
    get healthStatus() {
      return Alpine.store("console").healthStatus;
    },
    get toasts() {
      return Alpine.store("console").toasts;
    },
    init() {
      Alpine.store("console").checkHealth();
    },
    removeToast(id) {
      Alpine.store("console").removeToast(id);
    },
  };
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, options);
  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    payload = await response.json();
  } else {
    payload = await response.text();
  }
  if (!response.ok) {
    const message = payload?.message || payload?.detail || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload;
}

async function postMultipart(path, formData) {
  return apiFetch(path, { method: "POST", body: formData });
}

function rootStore() {
  return Alpine.store("console");
}

function dashboardPage() {
  return {
    data: {},
    async load() {
      const store = rootStore();
      store.setLoading(true);
      try {
        this.data = await apiFetch(`${API_V1}/system/dashboard`);
      } catch (error) {
        store.showToast("Dashboard", error.message);
      } finally {
        store.setLoading(false);
      }
    },
  };
}

function recognitionPage() {
  return {
    profile: "kyc",
    selectedFile: null,
    previewUrl: null,
    result: null,

    onImageSelected(event) {
      const file = event.target.files?.[0];
      this.selectedFile = file || null;
      this.previewUrl = file ? URL.createObjectURL(file) : null;
    },

    async runRecognition() {
      const store = rootStore();
      if (!this.selectedFile) return;

      const formData = new FormData();
      formData.append("profile", this.profile);
      formData.append("image", this.selectedFile);

      store.setLoading(true);
      try {
        this.result = await postMultipart(`${API_V1}/recognition`, formData);
        store.showToast("Recognition", `Detected ${this.result.face_count} face(s).`);
        this.$nextTick(() => this.drawDetections());
      } catch (error) {
        store.showToast("Recognition", error.message);
      } finally {
        store.setLoading(false);
      }
    },

    totalTimeMs() {
      const stages = this.result?.timings?.stages || {};
      return Object.values(stages).reduce((sum, value) => sum + value * 1000, 0).toFixed(1);
    },

    stageSummary() {
      const stages = this.result?.timings?.stages || {};
      return Object.entries(stages)
        .map(([name, seconds]) => `${name}: ${(seconds * 1000).toFixed(1)}ms`)
        .join(" · ");
    },

    drawDetections() {
      if (!this.previewUrl || !this.result?.faces?.length) return;
      const image = new Image();
      image.src = this.previewUrl;
      image.onload = () => {
        this.result.faces.forEach((face, index) => {
          const canvas = document.getElementById(`detect-canvas-${index}`);
          if (!canvas) return;
          const context = canvas.getContext("2d");
          const box = face.bounding_box;
          canvas.width = image.width;
          canvas.height = image.height;
          context.drawImage(image, 0, 0);
          context.strokeStyle = "#3d8bfd";
          context.lineWidth = Math.max(2, image.width / 300);
          context.strokeRect(box.x, box.y, box.width, box.height);
        });
      };
    },
  };
}

function enrollmentPage() {
  return {
    identityName: "",
    employeeId: "",
    selectedFile: null,
    previewUrl: null,
    result: null,

    onImageSelected(event) {
      const file = event.target.files?.[0];
      this.selectedFile = file || null;
      this.previewUrl = file ? URL.createObjectURL(file) : null;
    },

    async submitEnrollment() {
      const store = rootStore();
      if (!this.selectedFile) return;

      const formData = new FormData();
      formData.append("identity_name", this.identityName);
      formData.append("employee_id", this.employeeId);
      formData.append("image", this.selectedFile);

      store.setLoading(true);
      try {
        const payload = await postMultipart(`${API_V1}/enrollment`, formData);
        this.result = { success: true, ...payload };
        store.showToast("Enrollment", payload.message);
      } catch (error) {
        this.result = { success: false, message: error.message };
        store.showToast("Enrollment", error.message);
      } finally {
        store.setLoading(false);
      }
    },
  };
}

function galleryPage() {
  return {
    entries: [],

    async loadGallery() {
      const store = rootStore();
      store.setLoading(true);
      try {
        const payload = await apiFetch(`${API_V1}/gallery`);
        this.entries = payload.entries || [];
      } catch (error) {
        store.showToast("Gallery", error.message);
      } finally {
        store.setLoading(false);
      }
    },

    async deleteIdentity(identityId) {
      const store = rootStore();
      if (!confirm(`Delete identity ${identityId}?`)) return;
      store.setLoading(true);
      try {
        await apiFetch(`${API_V1}/enrollment/${encodeURIComponent(identityId)}`, {
          method: "DELETE",
        });
        store.showToast("Gallery", `Deleted ${identityId}.`);
        await this.loadGallery();
      } catch (error) {
        store.showToast("Gallery", error.message);
      } finally {
        store.setLoading(false);
      }
    },

    async rebuildGallery() {
      const store = rootStore();
      store.setLoading(true);
      try {
        const payload = await apiFetch(`${API_V1}/gallery/rebuild`, { method: "POST" });
        store.showToast("Gallery", payload.message);
        await this.loadGallery();
      } catch (error) {
        store.showToast("Gallery", error.message);
      } finally {
        store.setLoading(false);
      }
    },
  };
}

const PIPELINE_STAGE_ORDER = [
  ["scrfd", "Detection"],
  ["alignment", "Alignment"],
  ["assessment", "Assessment"],
  ["embedding", "Embedding"],
  ["search", "Search"],
  ["verification", "Verification"],
];

function pipelinePage() {
  return {
    profile: "kyc",
    selectedFile: null,
    stages: [],

    onImageSelected(event) {
      this.selectedFile = event.target.files?.[0] || null;
    },

    async runPipeline() {
      const store = rootStore();
      if (!this.selectedFile) return;

      const formData = new FormData();
      formData.append("profile", this.profile);
      formData.append("image", this.selectedFile);

      store.setLoading(true);
      try {
        const payload = await postMultipart(`${API_V1}/recognition`, formData);
        this.stages = this.buildStages(payload);
        store.showToast("Pipeline", "Pipeline execution completed.");
      } catch (error) {
        store.showToast("Pipeline", error.message);
      } finally {
        store.setLoading(false);
      }
    },

    buildStages(payload) {
      const timings = payload.timings?.stages || {};
      return PIPELINE_STAGE_ORDER.map(([key, label]) => {
        const seconds = timings[key] ?? 0;
        const completed = seconds > 0 || this.stageHasOutput(key, payload);
        return {
          name: key,
          label,
          status: completed ? "completed" : "skipped",
          timeMs: seconds * 1000,
          expanded: false,
          details: this.stageDetails(key, payload),
        };
      }).filter((stage) => stage.status === "completed");
    },

    stageHasOutput(key, payload) {
      if (key === "search") return Boolean(payload.metadata?.search_results);
      if (key === "verification") return Boolean(payload.metadata?.verification?.length);
      if (key === "assessment") return payload.faces?.some((face) => face.assessment);
      if (key === "embedding") return payload.faces?.some((face) => face.embedding);
      if (key === "alignment") return payload.faces?.some((face) => face.has_alignment);
      return payload.face_count > 0;
    },

    stageDetails(key, payload) {
      if (key === "search" && payload.metadata?.search_results) {
        return JSON.stringify(payload.metadata.search_results, null, 2);
      }
      if (key === "verification" && payload.metadata?.verification) {
        return JSON.stringify(payload.metadata.verification, null, 2);
      }
      if (key === "assessment") {
        return JSON.stringify(payload.faces?.map((face) => face.assessment) || [], null, 2);
      }
      if (key === "embedding") {
        return JSON.stringify(payload.faces?.map((face) => face.embedding) || [], null, 2);
      }
      return JSON.stringify({ face_count: payload.face_count, profile: payload.profile }, null, 2);
    },
  };
}

function benchmarkPage() {
  return {
    profile: "search",
    iterations: 1,
    selectedFile: null,
    timings: null,

    onImageSelected(event) {
      this.selectedFile = event.target.files?.[0] || null;
    },

    formatMs(value) {
      return `${Number(value || 0).toFixed(1)} ms`;
    },

    async runBenchmark() {
      const store = rootStore();
      if (!this.selectedFile) return;

      const formData = new FormData();
      formData.append("profile", this.profile);
      formData.append("iterations", String(this.iterations));
      formData.append("image", this.selectedFile);

      store.setLoading(true);
      try {
        const payload = await postMultipart(`${API_V1}/benchmark`, formData);
        this.timings = payload.timings;
        store.showToast("Benchmark", payload.message);
      } catch (error) {
        store.showToast("Benchmark", error.message);
      } finally {
        store.setLoading(false);
      }
    },
  };
}

function configurationPage() {
  return {
    files: {},
    async load() {
      const store = rootStore();
      store.setLoading(true);
      try {
        this.files = await apiFetch(`${API_V1}/configuration/files`);
      } catch (error) {
        store.showToast("Configuration", error.message);
      } finally {
        store.setLoading(false);
      }
    },
  };
}

window.consoleApp = consoleApp;
window.dashboardPage = dashboardPage;
window.recognitionPage = recognitionPage;
window.enrollmentPage = enrollmentPage;
window.galleryPage = galleryPage;
window.pipelinePage = pipelinePage;
window.benchmarkPage = benchmarkPage;
window.configurationPage = configurationPage;
