export type Job = {
  id: string;
  kind: string;
  project_id?: string;
  status: 'queued' | 'running' | 'success' | 'error' | 'cancelled';
  progress: number;
  current_step: string;
  error?: string;
  result?: Record<string, unknown>;
};

export type Project = {
  id: string;
  name: string;
  target_type: string;
  status: string;
  path: string;
  created_at: string;
  updated_at: string;
  last_step?: string;
};

const API = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    let detail = await res.text();
    try {
      detail = JSON.stringify(JSON.parse(detail).detail);
    } catch {
      // Keep plain text detail.
    }
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  projects: () => request<Project[]>('/projects'),
  createProject: (name: string, target_type: string) =>
    request<Project>('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, target_type })
    }),
  project: (id: string) => request<Project>(`/projects/${id}`),
  upload: (id: string, kind: 'video' | 'capturepack' | 'training-result', file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<Record<string, unknown>>(`/projects/${id}/import/${kind}`, { method: 'POST', body: form });
  },
  analyzeQuality: (id: string) => request<Job>(`/projects/${id}/quality/analyze`, { method: 'POST' }),
  qualityReport: (id: string) => request<Record<string, any>>(`/projects/${id}/quality/report`),
  extractFrames: (id: string, settings: Record<string, unknown>) =>
    request<Job>(`/projects/${id}/preprocess/extract-frames`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    }),
  frameIndex: (id: string) => request<Record<string, any>>(`/projects/${id}/preprocess/frame-index`),
  runColmap: (id: string) => request<Job>(`/projects/${id}/preprocess/run-colmap`, { method: 'POST' }),
  poseReport: (id: string) => request<Record<string, any>>(`/projects/${id}/preprocess/pose-report`),
  trainingStatus: (id: string) => request<Record<string, any>>(`/projects/${id}/training/status`),
  localTraining: (id: string, payload: Record<string, unknown>) =>
    request<Job>(`/projects/${id}/training/local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }),
  colabPackage: (id: string, payload: Record<string, unknown>) =>
    request<Job>(`/projects/${id}/training/create-colab-package`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }),
  previewStatus: (id: string) => request<Record<string, any>>(`/projects/${id}/preview/status`),
  buildPreview: (id: string) => request<Record<string, unknown>>(`/projects/${id}/preview/build`, { method: 'POST' }),
  previewConfig: (id: string) => request<Record<string, any>>(`/projects/${id}/preview/config`),
  export: (id: string, export_type: string) =>
    request<Job>(`/projects/${id}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ export_type })
    }),
  exports: (id: string) => request<Array<Record<string, any>>>(`/projects/${id}/export/list`),
  job: (id: string) => request<Job>(`/jobs/${id}`),
  jobLogs: async (id: string) => (await fetch(`${API}/jobs/${id}/logs`)).text()
};

export const downloadUrl = (path: string) => `${API}${path}`;

