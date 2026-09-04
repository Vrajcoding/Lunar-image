import axios from "axios";

// Access base URL from environment or fallback to localhost:8000
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function registerImages(sourceFile, referenceFile, sourceLabelFile, referenceLabelFile) {
  const formData = new FormData();
  formData.append("source", sourceFile);
  formData.append("reference", referenceFile);
  // Optional detached label (e.g. a PDS4 .xml next to a .img) — a plain
  // PNG/TIFF upload sends neither and the backend behaves exactly as before.
  if (sourceLabelFile) formData.append("source_label", sourceLabelFile);
  if (referenceLabelFile) formData.append("reference_label", referenceLabelFile);

  const res = await axios.post(`${BASE_URL}/api/register`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // { job_id, status }
}

export async function getStatus(jobId) {
  const res = await axios.get(`${BASE_URL}/api/status/${jobId}`);
  return res.data; // { job_id, status }
}

export async function getResult(jobId) {
  const res = await axios.get(`${BASE_URL}/api/result/${jobId}`);
  const data = res.data;
  
  // Format URLs to absolute if they start with /
  if (data.registered_image_url && data.registered_image_url.startsWith("/")) {
    data.registered_image_url = `${BASE_URL}${data.registered_image_url}`;
  }
  if (data.match_points_url && data.match_points_url.startsWith("/")) {
    data.match_points_url = `${BASE_URL}${data.match_points_url}`;
  }
  return data;
}

export function downloadUrl(jobId, fileType) {
  return `${BASE_URL}/api/download/${jobId}/${fileType}`;
}
