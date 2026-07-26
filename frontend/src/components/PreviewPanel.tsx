import { RefreshCw } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api, Project } from '../api/client';
import SplatViewer from '../viewer/SplatViewer';

export default function PreviewPanel({ project, onRefresh }: { project: Project; onRefresh: () => Promise<void> }) {
  const [status, setStatus] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState('');

  async function load() {
    try {
      setStatus(await api.previewStatus(project.id));
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => { load(); }, [project.id]);

  async function build() {
    try {
      await api.buildPreview(project.id);
      await load();
      await onRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const ready = Boolean(status?.ready);
  return (
    <div className="stack">
      {error && <div className="alert error">{error}</div>}
      <div className="toolbar">
        <button onClick={build}><RefreshCw size={18} /> Build From Latest Training</button>
        <span className="muted">{ready ? status?.config?.model_file : status?.message}</span>
      </div>
      <div className="viewer-wrap">
        {ready ? <SplatViewer modelUrl={`/api/projects/${project.id}/preview/model`} modelType={status?.config?.model_type || 'ply'} /> : <div className="empty">Import a trained .ply/.splat result to preview.</div>}
      </div>
    </div>
  );
}

