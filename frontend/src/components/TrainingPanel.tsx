import { Cloud, Cpu, Download } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api, downloadUrl, Job, Project } from '../api/client';
import JobProgress from './JobProgress';

export default function TrainingPanel({ project, onRefresh }: { project: Project; onRefresh: () => Promise<void> }) {
  const [preset, setPreset] = useState('draft');
  const [scene, setScene] = useState('object');
  const [status, setStatus] = useState<Record<string, any> | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');

  async function load() {
    setStatus(await api.trainingStatus(project.id));
  }

  useEffect(() => { load().catch(() => undefined); }, [project.id]);

  async function startLocal() {
    setError('');
    try {
      setJob(await api.localTraining(project.id, { mode: 'local', preset, scene_type: scene }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function createColab() {
    setError('');
    try {
      setJob(await api.colabPackage(project.id, { preset, scene_type: scene }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="stack">
      {error && <div className="alert error">{error}</div>}
      <div className="panel-grid">
        <div className="panel">
          <h3>Training Config</h3>
          <label>Preset<select value={preset} onChange={(e) => setPreset(e.target.value)}>
            <option value="draft">Draft</option>
            <option value="balanced">Balanced</option>
            <option value="high">High Quality</option>
          </select></label>
          <label>Scene Type<select value={scene} onChange={(e) => setScene(e.target.value)}>
            <option value="object">Object</option>
            <option value="room">Room / Interior</option>
            <option value="outdoor">Outdoor / Large Scene</option>
          </select></label>
        </div>
        <div className="panel">
          <h3>Local GPU</h3>
          <p className="muted">{status?.trainer?.exists ? status.trainer.path : 'Trainer not configured.'}</p>
          <button onClick={startLocal}><Cpu size={18} /> Start Local Training</button>
        </div>
        <div className="panel">
          <h3>Colab GPU Package</h3>
          <p className="muted">Creates dataset.zip for CUDA GPU runtime. TPU is not supported.</p>
          <button onClick={createColab}><Cloud size={18} /> Create Colab Package</button>
          <a className="button-link" href={downloadUrl(`/projects/${project.id}/training/download-colab-package`)}>
            <Download size={18} /> Download Dataset Zip
          </a>
        </div>
      </div>
      <JobProgress job={job} onDone={async () => { await load(); await onRefresh(); }} />
    </div>
  );
}

