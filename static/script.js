const video = document.getElementById('video');

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    // For some browsers, wait for metadata to know width/height
    await new Promise((res) => video.onloadedmetadata = res);
  } catch (err) {
    console.error("Camera error:", err.name, err.message);
    // optional: show a user-friendly UI message
  }
}

async function captureFrame() {
  // ensure video has size
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(video, 0, 0, w, h);
  return await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
}

// expose globally for inline scripts returned by HTMX
window.captureFrame = captureFrame;
window.startCamera = startCamera;

// start immediately
startCamera();