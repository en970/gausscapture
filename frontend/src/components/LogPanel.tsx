import { useState } from 'react';
import { api } from '../api/client';

export default function LogPanel() {
  const [jobId, setJobId] = useState('');
  const [logs, setLogs] = useState('');

  async function load() {
    if (!jobId.trim()) return;
    setLogs(await api.jobLogs(jobId.trim()));
  }

  return (
    <div className="panel wide">
      <h3>Job Logs</h3>
      <div className="control-row">
        <input value={jobId} onChange={(e) => setJobId(e.target.value)} placeholder="Job id" />
        <button onClick={load}>Load Logs</button>
        <button onClick={() => navigator.clipboard.writeText(logs)}>Copy Log</button>
      </div>
      <pre className="log-view">{logs || 'Run a job to see live logs in each panel, or paste a job id here.'}</pre>
    </div>
  );
}

