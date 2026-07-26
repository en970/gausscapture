import { Camera, Images } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api, Job, Project } from '../api/client';
import JobProgress from './JobProgress';

export default function PreprocessPanel({ project, onRefresh }: { project: Project; onRefresh: () => Promise<void> }) {
  const [settings, setSettings] = useState({
    target_fps: 2,
    max_frames: 600,
    resize_max_side: 1920,
    blur_filter: true,
    duplicate_filter: true
  });
  const [job, setJob] = useState<Job | null>(null);
  const [frameIndex, setFrameIndex] = useState<Record<string, any> | null>(null);
  const [pose, setPose] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState('');

  async function load() {
    try { setFrameIndex(await api.frameIndex(project.id)); } catch { setFrameIndex(null); }
    try { setPose(await api.poseReport(project.id)); } catch { setPose(null); }
  }

  useEffect(() => { load(); }, [project.id]);

  async function extract() {
    setError('');
    try {
      setJob(await api.extractFrames(project.id, settings));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function colmap() {
    setError('');
    try {
      setJob(await api.runColmap(project.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="stack">
      {error && <div className="alert error">{error}</div>}
      <div className="panel-grid">
        <div className="panel wide">
          <h3>Frame Extraction</h3>
          <div className="control-row">
            <label>Target FPS<select value={settings.target_fps} onChange={(e) => setSettings({ ...settings, target_fps: Number(e.target.value) })}>
              {[1, 2, 5, 10].map((v) => <option key={v} value={v}>{v} fps</option>)}
            </select></label>
            <label>Max Frames<select value={settings.max_frames} onChange={(e) => setSettings({ ...settings, max_frames: Number(e.target.value) })}>
              {[100, 300, 600, 1000].map((v) => <option key={v} value={v}>{v}</option>)}
            </select></label>
            <label>Resize<select value={settings.resize_max_side} onChange={(e) => setSettings({ ...settings, resize_max_side: Number(e.target.value) })}>
              {[1024, 1280, 1920].map((v) => <option key={v} value={v}>{v} max side</option>)}
            </select></label>
            <label className="check"><input type="checkbox" checked={settings.blur_filter} onChange={(e) => setSettings({ ...settings, blur_filter: e.target.checked })} /> Blur filter</label>
            <label className="check"><input type="checkbox" checked={settings.duplicate_filter} onChange={(e) => setSettings({ ...settings, duplicate_filter: e.target.checked })} /> Duplicate filter</label>
          </div>
          <button className="primary" onClick={extract}><Images size={18} /> Extract Frames</button>
        </div>
        <div className="panel">
          <h3>Frames</h3>
          <p><strong>{frameIndex?.frames_used ?? 0}</strong> usable frames</p>
          <p>{frameIndex?.frames_skipped_blur ?? 0} blur skipped</p>
          <p>{frameIndex?.frames_skipped_duplicate ?? 0} duplicate skipped</p>
        </div>
        <div className="panel">
          <h3>Camera Poses</h3>
          <p>{pose ? `${pose.images_registered}/${pose.images_total} registered` : 'No pose report yet.'}</p>
          <button onClick={colmap}><Camera size={18} /> Run COLMAP</button>
        </div>
      </div>
      <JobProgress job={job} onDone={async () => { await load(); await onRefresh(); }} />
    </div>
  );
}

