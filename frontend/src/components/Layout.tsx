import { Box, FolderOpen, Gauge, HelpCircle, Settings } from 'lucide-react';
import type { ReactNode } from 'react';
import { Project } from '../api/client';

type Props = {
  projects: Project[];
  active: Project | null;
  onSelect: (project: Project) => void;
  onDashboard: () => void;
  children: ReactNode;
};

export default function Layout({ projects, active, onSelect, onDashboard, children }: Props) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={onDashboard}>
          <Box size={22} />
          <span>GaussCapture</span>
        </button>
        <nav className="nav">
          <button className={!active ? 'active' : ''} onClick={onDashboard}>
            <Gauge size={18} /> Dashboard
          </button>
          <div className="nav-label">Projects</div>
          {projects.map((project) => (
            <button key={project.id} className={active?.id === project.id ? 'active' : ''} onClick={() => onSelect(project)}>
              <FolderOpen size={18} /> {project.name}
            </button>
          ))}
          <button disabled>
            <Settings size={18} /> Settings
          </button>
          <button disabled>
            <HelpCircle size={18} /> Help
          </button>
        </nav>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
