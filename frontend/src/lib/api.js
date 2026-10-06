// src/lib/api.js
import axios from 'axios'

const BASE = '/api'

export const api = {
  // Health
  health: () => axios.get(`${BASE}/`),

  // Subjects
  listSubjects: () => axios.get(`${BASE}/subjects`),
  getNight: (id) => axios.get(`${BASE}/subjects/${id}/night`),
  getTwinState: (id, start = 0, end = null) => {
    const params = { start_minute: start }
    if (end !== null) params.end_minute = end
    return axios.get(`${BASE}/subjects/${id}/twin-state`, { params })
  },
  predictAt: (id, t) => axios.get(`${BASE}/subjects/${id}/predict`, { params: { t } }),

  // Simulate
  simulate: (body) => axios.post(`${BASE}/simulate`, body),

  // Validation
  getValidation: () => axios.get(`${BASE}/validation`),
  getDatasetInfo: () => axios.get(`${BASE}/dataset-info`),
}

// Risk score → colour mapping
export function riskColor(score) {
  if (score < 0.35) return 'var(--risk-low)'
  if (score < 0.65) return 'var(--risk-medium)'
  return 'var(--risk-high)'
}

export function riskLabel(score) {
  if (score < 0.35) return 'Low'
  if (score < 0.65) return 'Medium'
  return 'High'
}

export function riskClass(score) {
  if (score < 0.35) return 'risk-low'
  if (score < 0.65) return 'risk-medium'
  return 'risk-high'
}

export function severityClass(severity) {
  return `severity-${severity || 'normal'}`
}
