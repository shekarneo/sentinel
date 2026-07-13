function applyCanvasRenderSpec(canvas, renderSpec) {
  if (!canvas) return;

  const context = canvas.getContext("2d");
  if (!renderSpec?.enabled) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  (renderSpec.faces || []).forEach((face) => {
    const box = face.bounding_box || {};
    context.strokeStyle = face.stroke_color || "#3d8bfd";
    context.lineWidth = Math.max(2, canvas.width / 300);
    context.strokeRect(box.x, box.y, box.width, box.height);
    const label = face.label || face.identity_id || "unknown";
    context.fillStyle = "rgba(13, 17, 23, 0.75)";
    context.fillRect(box.x, box.y - 20, Math.max(label.length * 7, 80), 20);
    context.fillStyle = "#f0f6fc";
    context.font = "12px sans-serif";
    context.fillText(label, box.x + 4, box.y - 6);
  });
}

window.applyCanvasRenderSpec = applyCanvasRenderSpec;
