import axios from "axios";

const API_BASE = "";

export const registerImages = async (
  sourceFile,
  referenceFile,
  sourceLabelFile = null,
  referenceLabelFile = null,
  band = null,
  mode = "auto",
  sourceSensor = "Auto / Unknown",
  referenceSensor = "Auto / Unknown"
) => {
  const formData = new FormData();
  formData.append("source", sourceFile);
  formData.append("reference", referenceFile);

  if (sourceLabelFile) {
    formData.append("source_label", sourceLabelFile);
  }
  if (referenceLabelFile) {
    formData.append("reference_label", referenceLabelFile);
  }
  if (band !== null && band !== undefined) {
    formData.append("band", band);
  }
  if (mode) {
    formData.append("mode", mode);
  }
  if (sourceSensor) {
    formData.append("source_sensor", sourceSensor);
  }
  if (referenceSensor) {
    formData.append("reference_sensor", referenceSensor);
  }

  const res = await axios.post(`${API_BASE}/api/register`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return res.data;
};

export const getStatus = async (jobId) => {
  const res = await axios.get(`${API_BASE}/api/status/${jobId}`);
  return res.data;
};

export const getResult = async (jobId) => {
  const res = await axios.get(`${API_BASE}/api/result/${jobId}`);
  return res.data;
};
