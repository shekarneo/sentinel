const API_V1 = "/api/v1";

function liveCameraPage() {
  return {
    sessionId: null,
    sessionActive: false,
    status: "idle",
    transport: "websocket",
    socket: null,
    config: {
      camera_index: 0,
      resolution_width: 640,
      resolution_height: 480,
      target_fps: 15,
      pipeline_profile: "surveillance",
      overlay_enabled: true,
      submission_interval_ms: 500,
      recognition_policy: {
        policy_type: "every_n_frames",
        frame_interval: 3,
      },
    },
    metrics: {},
    overlay: null,
    renderSpec: null,
    timeline: [],
    processedFrames: 0,
    stream: null,
    captureTimer: null,
    video: null,
    canvas: null,

    get overlayFaces() {
      return this.overlay?.faces || [];
    },

    async init() {
      this.video = document.getElementById("live-video");
      this.canvas = document.getElementById("live-overlay-canvas");
      await this.refreshStatus();
    },

    async refreshStatus() {
      try {
        const payload = await fetch(`${API_V1}/live/status`).then((r) => r.json());
        if (payload.active && payload.session) {
          this.sessionId = payload.session.id;
          this.sessionActive = payload.session.status !== "stopped";
          this.status = payload.session.status;
          this.metrics = payload.session.metrics || {};
          this.timeline = payload.session.timeline || [];
          this.overlay = payload.session.last_overlay;
          this.processedFrames = payload.session.processed_frames || 0;
          this.config = {
            ...this.config,
            ...payload.session.config,
          };
        }
      } catch (_error) {
        // Ignore status errors on first load.
      }
    },

    async startSession() {
      const store = Alpine.store("console");
      store.setLoading(true);
      try {
        await this.openCamera();
        const response = await fetch(`${API_V1}/live/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ config: this.config }),
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const session = await response.json();
        this.sessionId = session.id;
        this.sessionActive = true;
        this.status = session.status;
        this.metrics = session.metrics || {};
        this.timeline = session.timeline || [];
        this.processedFrames = 0;
        await this.connectWebSocket();
        this.startCaptureLoop();
        store.showToast("Live Camera", "Session started.");
      } catch (error) {
        this.disconnectWebSocket();
        this.stopCamera();
        store.showToast("Live Camera", error.message || "Failed to start session.");
      } finally {
        store.setLoading(false);
      }
    },

    async stopSession() {
      if (!this.sessionId) return;
      this.stopCaptureLoop();
      this.disconnectWebSocket();
      this.stopCamera();
      await fetch(`${API_V1}/live/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: this.sessionId }),
      });
      this.sessionActive = false;
      this.status = "stopped";
      this.sessionId = null;
      applyCanvasRenderSpec(this.canvas, { enabled: false });
    },

    async pauseSession() {
      if (!this.sessionId) return;
      this.stopCaptureLoop();
      const response = await fetch(`${API_V1}/live/pause`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: this.sessionId }),
      });
      const session = await response.json();
      this.status = session.status;
    },

    async resumeSession() {
      if (!this.sessionId) return;
      const response = await fetch(`${API_V1}/live/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: this.sessionId }),
      });
      const session = await response.json();
      this.status = session.status;
      this.startCaptureLoop();
    },

    async connectWebSocket() {
      this.disconnectWebSocket();
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const url = `${protocol}://${window.location.host}${API_V1}/live/ws`;
      this.socket = new WebSocket(url);
      this.socket.binaryType = "arraybuffer";
      await new Promise((resolve, reject) => {
        this.socket.onopen = () => resolve();
        this.socket.onerror = () => reject(new Error("WebSocket connection failed."));
      });
      this.socket.onmessage = (event) => this.handleSocketMessage(event);
    },

    disconnectWebSocket() {
      if (this.socket) {
        this.socket.close();
        this.socket = null;
      }
    },

    handleSocketMessage(event) {
      const payload = JSON.parse(event.data);
      if (payload.type === "error") {
        Alpine.store("console").showToast("Live Camera", payload.message);
        return;
      }
      if (payload.type !== "frame_result") return;
      this.applyFrameResult(payload);
    },

    applyFrameResult(payload) {
      this.metrics = payload.metrics || {};
      this.overlay = payload.overlay || null;
      this.renderSpec = payload.render_spec || null;
      if (this.renderSpec) {
        applyCanvasRenderSpec(this.canvas, this.renderSpec);
      }
    },

    async openCamera() {
      const constraints = {
        video: {
          width: { ideal: this.config.resolution_width },
          height: { ideal: this.config.resolution_height },
          frameRate: { ideal: this.config.target_fps },
        },
      };
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      await this.video.play();
      this.resizeOverlayCanvas();
    },

    stopCamera() {
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
      if (this.video) {
        this.video.srcObject = null;
      }
    },

    startCaptureLoop() {
      this.stopCaptureLoop();
      const interval = Math.max(this.config.submission_interval_ms || 500, 50);
      this.captureTimer = window.setInterval(() => this.captureFrame(), interval);
    },

    stopCaptureLoop() {
      if (this.captureTimer) {
        window.clearInterval(this.captureTimer);
        this.captureTimer = null;
      }
    },

    async captureFrame() {
      if (!this.sessionId || this.status !== "running" || !this.video || this.video.readyState < 2) {
        return;
      }

      const captureCanvas = document.createElement("canvas");
      captureCanvas.width = this.video.videoWidth || this.config.resolution_width;
      captureCanvas.height = this.video.videoHeight || this.config.resolution_height;
      const context = captureCanvas.getContext("2d");
      context.drawImage(this.video, 0, 0, captureCanvas.width, captureCanvas.height);

      const blob = await new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.85));
      if (!blob) return;

      if (this.transport === "websocket" && this.socket?.readyState === WebSocket.OPEN) {
        const header = JSON.stringify({
          type: "frame",
          session_id: this.sessionId,
          force: false,
        });
        const headerBytes = new TextEncoder().encode(`${header}\n`);
        const frameBytes = new Uint8Array(await blob.arrayBuffer());
        const payload = new Uint8Array(headerBytes.length + frameBytes.length);
        payload.set(headerBytes, 0);
        payload.set(frameBytes, headerBytes.length);
        this.socket.send(payload.buffer);
        await this.refreshStatus();
        return;
      }

      const formData = new FormData();
      formData.append("session_id", this.sessionId);
      formData.append("image", blob, "frame.jpg");
      formData.append("force", "false");

      const response = await fetch(`${API_V1}/live/frame`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) return;
      this.applyFrameResult(await response.json());
      await this.refreshStatus();
    },

    resizeOverlayCanvas() {
      if (!this.canvas || !this.video) return;
      const width = this.video.videoWidth || this.config.resolution_width;
      const height = this.video.videoHeight || this.config.resolution_height;
      this.canvas.width = width;
      this.canvas.height = height;
    },

    formatWhen(value) {
      return new Date(value).toLocaleString();
    },
  };
}

window.liveCameraPage = liveCameraPage;
