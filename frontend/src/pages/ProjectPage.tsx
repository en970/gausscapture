import { useState } from 'react';
import { Project } from '../api/client';
import CapturePackImport from '../components/CapturePackImport';
import ExportPanel from '../components/ExportPanel';
import LogPanel from '../components/LogPanel';
import PreprocessPanel from '../components/PreprocessPanel';
import PreviewPanel from '../components/PreviewPanel';
import QualityReport from '../components/QualityReport';
import TrainingPanel from '../components/TrainingPanel';

const tabs = ['Import', 'Quality', 'Preprocess', 'Train', 'Preview', 'Export', 'Logs'] as const;
type Tab = typeof tabs[number];

export default function ProjectPage({ project, onRefresh }: { project: Project; onRefresh: () => Promise<void> }) {
  const [tab, setTab] = useState<Tab>('Import');

  return (
    <div className="page">
      <header className="project-header">
        <div>
          <h1>{project.name}</h1>
          <p>{project.path}</p>
        </div>
        <div className="status-card">
          <span className="badge">{project.status}</span>
          <span>{project.last_step}</span>
        </div>
      </header>
      <div className="tabs">
        {tabs.map((item) => <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>)}
      </div>
      {tab === 'Import' && <CapturePackImport project={project} onRefresh={onRefresh} />}
      {tab === 'Quality' && <QualityReport project={project} onRefresh={onRefresh} />}
      {tab === 'Preprocess' && <PreprocessPanel project={project} onRefresh={onRefresh} />}
      {tab === 'Train' && <TrainingPanel project={project} onRefresh={onRefresh} />}
      {tab === 'Preview' && <PreviewPanel project={project} onRefresh={onRefresh} />}
      {tab === 'Export' && <ExportPanel project={project} onRefresh={onRefresh} />}
      {tab === 'Logs' && <LogPanel />}
    </div>
  );
}

