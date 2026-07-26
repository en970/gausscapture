import { useEffect, useState } from 'react';
import { api, Job } from '../api/client';

type Props = {
  job: Job | null;
  onDone?: () => void;
};

export default function JobProgress({ job, onDone }: Props) {
  const [current, setCurrent] = useState<Job | null>(job);
  const [logs, setLogs] = useState('');

  useEffect(() => {
    setCurrent(job);
  }, [job?.id]);

  useEffect(() => {
    if (!current?.id || ['success', 'error', 'cancelled'].includes(current.status)) return;
    const timer = window.setInterval(async () => {
      const next = await api.job(current.id);
      setCurrent(next);
      setLogs(await api.jobLogs(current.id));
      if (next.status === 'success' || next.status === 'error') {
        onDone?.();
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [current?.id, current?.status]);

  if (!current) return null;
  return (
    <div className="job-box">
      <div className="job-line">
        <strong>{current.kind}</strong>
        <span>{current.status}</span>
        <span>{current.progress}%</span>
      </div>
      <div className="progress">
        <div style={{ width: `${current.progress}%` }} />
      </div>
      <div className="muted">{current.current_step}</div>
      {current.error && <div className="alert error">{current.error}</div>}
      {logs && <pre className="log-mini">{logs}</pre>}
    </div>
  );
}

