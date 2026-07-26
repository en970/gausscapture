import { useCallback, useEffect, useState } from 'react';
import { api, Project } from './api/client';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import ProjectPage from './pages/ProjectPage';

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [active, setActive] = useState<Project | null>(null);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    const list = await api.projects();
    setProjects(list);
    if (active) {
      const next = list.find((p) => p.id === active.id);
      if (next) setActive(next);
    }
  }, [active]);

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  async function createProject(name: string, target: string) {
    const project = await api.createProject(name, target);
    await refresh();
    setActive(project);
  }

  return (
    <Layout projects={projects} active={active} onSelect={setActive} onDashboard={() => setActive(null)}>
      {error && <div className="alert error">{error}</div>}
      {active ? (
        <ProjectPage project={active} onRefresh={refresh} />
      ) : (
        <Dashboard projects={projects} onCreate={createProject} onSelect={setActive} />
      )}
    </Layout>
  );
}

