import { ArrowRight, Clock } from 'lucide-react';
import { Project } from '../api/client';
import ProjectCreate from '../components/ProjectCreate';

type Props = {
  projects: Project[];
  onCreate: (name: string, target: string) => Promise<void>;
  onSelect: (project: Project) => void;
};

export default function Dashboard({ projects, onCreate, onSelect }: Props) {
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>GaussCapture MVP</h1>
          <p>Local CapturePack import, preprocessing, 3DGS training wrapper, preview, and export.</p>
        </div>
      </header>
      <ProjectCreate onCreate={onCreate} />
      <section className="project-list">
        <h2>Recent Projects</h2>
        {projects.length === 0 && <div className="empty">No projects yet. Create one and import a phone video.</div>}
        {projects.map((project) => (
          <button className="project-row" key={project.id} onClick={() => onSelect(project)}>
            <div>
              <strong>{project.name}</strong>
              <span><Clock size={15} /> {project.status} · {project.last_step || 'Ready'}</span>
            </div>
            <ArrowRight size={18} />
          </button>
        ))}
      </section>
    </div>
  );
}

