function initRecognitionPanels(root = document) {
  root.querySelectorAll("[data-crop-face]").forEach((canvas) => {
    drawFaceCrop(canvas);
  });
  root.querySelectorAll("[data-detect-canvas]").forEach((canvas) => {
    drawDetectionCanvas(canvas);
  });
  root.querySelectorAll("[data-stage-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.getAttribute("data-stage-toggle");
      const panel = root.querySelector(`#${targetId}`);
      if (panel) {
        panel.classList.toggle("d-none");
      }
    });
  });
  root.querySelectorAll("[data-debug-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const panel = root.querySelector("#recognition-debug-panel");
      if (panel) panel.classList.toggle("d-none");
    });
  });
}

function drawDetectionCanvas(canvas) {
  const imageSrc = canvas.getAttribute("data-source-image");
  const boxes = JSON.parse(canvas.getAttribute("data-boxes") || "[]");
  if (!imageSrc) return;

  const image = new Image();
  image.onload = () => {
    const context = canvas.getContext("2d");
    canvas.width = image.width;
    canvas.height = image.height;
    context.drawImage(image, 0, 0);
    context.strokeStyle = "#3d8bfd";
    context.lineWidth = Math.max(2, image.width / 300);
    boxes.forEach((box) => {
      context.strokeRect(box.x, box.y, box.width, box.height);
    });
  };
  image.src = imageSrc;
}

function drawFaceCrop(canvas) {
  const imageSrc = canvas.getAttribute("data-source-image");
  const box = JSON.parse(canvas.getAttribute("data-box") || "{}");
  if (!imageSrc || !box.width) return;

  const image = new Image();
  image.onload = () => {
    const context = canvas.getContext("2d");
    const size = 112;
    canvas.width = size;
    canvas.height = size;
    context.drawImage(image, box.x, box.y, box.width, box.height, 0, 0, size, size);
  };
  image.src = imageSrc;
}

document.body.addEventListener("htmx:afterSwap", (event) => {
  if (event.target?.id === "recognition-results") {
    initRecognitionPanels(event.target);

    const payloadNode = event.target.querySelector("#recognition-payload");
    if (payloadNode) {
      const payload = JSON.parse(payloadNode.textContent || "{}");
      const totalMs = payload.timeline?.total_ms || 0;
      document.body.dispatchEvent(
        new CustomEvent("recognition:completed", {
          detail: {
            id: payload.run_id,
            timestamp: payload.timestamp,
            profile: payload.result?.profile,
            face_count: payload.result?.face_count,
            total_ms: totalMs,
            payload,
          },
        }),
      );
    }
  }
});

document.addEventListener("DOMContentLoaded", () => {
  initRecognitionPanels(document.getElementById("recognition-results"));
});

window.initRecognitionPanels = initRecognitionPanels;
