import { Archive, Download } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api, downloadUrl, Job, Project } from '../api/client';
import JobProgress from './JobProgress';

const exportTypes = [
  ['raw', 'Raw Gaussian Splat'],
  ['web', 'Web Viewer Bundle'],
  ['blender', 'Blender Export'],
  ['unity', 'Unity Package'],
  ['proxy_mesh', 'Proxy Mesh']
];

export default function ExportPanel({ project, onRefresh }: { project: Project; onRefresh: () => Promise<void> }) {
  const [job, setJob] = useState<Job | null>(null);
  const [exports, setExports] = useState<Array<Record<string, any>>>([]);
  const [error, setError] = useState('');

  async function load() {
    try { setExports(await api.exports(project.id)); } catch { setExports([]); }
  }

  useEffect(() => { load(); }, [project.id]);

  async function create(type: string) {
    setError('');
    try {
      setJob(await api.export(project.id, type));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="stack">
      {error && <div className="alert error">{error}</div>}
      <div className="panel-grid">
        {exportTypes.map(([type, label]) => (
          <div className="panel" key={type}>
            <h3>{label}</h3>
            <button onClick={() => create(type)}><Archive size={18} /> Create Export</button>
          </div>
        ))}
      </div>
      <JobProgress job={job} onDone={async () => { await load(); await onRefresh(); }} />
      <div className="panel wide">
        <h3>Downloads</h3>
        {exports.length === 0 && <p className="muted">No exports created yet.</p>}
        {exports.map((item) => (
          <a className="download-row" key={item.id} href={downloadUrl(`/projects/${project.id}/export/download/${item.id}`)}>
            <Download size={18} /> {item.name} <span>{Math.round(item.size_bytes / 1024)} KB</span>
          </a>
        ))}
      </div>
    </div>
  );
}

