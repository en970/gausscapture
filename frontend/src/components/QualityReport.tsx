import { Activity } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api, Job, Project } from '../api/client';
import JobProgress from './JobProgress';

export default function QualityReport({ project, onRefresh }: { project: Project; onRefresh: () => Promise<void> }) {
  const [report, setReport] = useState<Record<string, any> | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');

  async function load() {
    try {
      setReport(await api.qualityReport(project.id));
      setError('');
    } catch {
      setReport(null);
    }
  }

  useEffect(() => {
    load();
  }, [project.id]);

  async function analyze() {
    setJob(await api.analyzeQuality(project.id));
  }

  return (
    <div className="stack">
      <button className="primary" onClick={analyze}>
        <Activity size={18} /> Analyze Quality
      </button>
      <JobProgress job={job} onDone={async () => { await load(); await onRefresh(); }} />
      {error && <div className="alert error">{error}</div>}
      {report ? <ReportView project={project} report={report} /> : <div className="empty">No quality report yet.</div>}
    </div>
  );
}

function ReportView({ project, report }: { project: Project; report: Record<string, any> }) {
  return (
    <div className="panel-grid">
      <div className={`score ${report.status}`}>
        <span>{report.overall_score}</span>
        <strong>{report.status}</strong>
      </div>
      <div className="panel">
        <h3>Video</h3>
        <p>{report.video?.resolution} at {Number(report.video?.fps || 0).toFixed(2)} fps</p>
        <p>{Number(report.video?.duration_sec || 0).toFixed(1)} seconds</p>
      </div>
      <Metric title="Brightness" data={report.brightness} />
      <Metric title="Blur" data={report.blur} />
      <Metric title="Motion" data={report.motion} />
      <div className="panel wide">
        <h3>Warnings</h3>
        <ul>{(report.warnings || []).map((w: string) => <li key={w}>{w}</li>)}</ul>
        <h3>Recommendations</h3>
        <ul>{(report.recommendations || []).map((w: string) => <li key={w}>{w}</li>)}</ul>
      </div>
      <div className="thumb-row wide">
        {(report.thumbnails || []).map((thumb: string) => (
          <img key={thumb} src={`/api/projects/${project.id}/files/${thumb}`} />
        ))}
      </div>
    </div>
  );
}

function Metric({ title, data }: { title: string; data: Record<string, any> }) {
  return (
    <div className="panel">
      <h3>{title}</h3>
      {Object.entries(data || {}).map(([key, value]) => (
        <p key={key}><span className="muted">{key}</span> {typeof value === 'number' ? value.toFixed(3) : String(value)}</p>
      ))}
    </div>
  );
}

