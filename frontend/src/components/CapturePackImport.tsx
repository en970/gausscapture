import { UploadCloud } from 'lucide-react';
import { useState } from 'react';
import { api, Job, Project } from '../api/client';

type Props = {
  project: Project;
  onRefresh: () => Promise<void>;
};

export default function CapturePackImport({ project, onRefresh }: Props) {
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  async function upload(kind: 'video' | 'capturepack' | 'training-result', file?: File) {
    if (!file) return;
    setBusy(true);
    setMessage('');
    try {
      const result = await api.upload(project.id, kind, file);
      await onRefresh();
      setMessage(JSON.stringify(result, null, 2));
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel-grid">
      <UploadBox title="Video Import" accept=".mp4,.mov,.m4v,.avi,.mkv" disabled={busy} onFile={(file) => upload('video', file)} />
      <UploadBox title="CapturePack Import" accept=".capturepack,.zip" disabled={busy} onFile={(file) => upload('capturepack', file)} />
      <UploadBox title="Training Result Import" accept=".zip,.ply,.splat,.ksplat" disabled={busy} onFile={(file) => upload('training-result', file)} />
      <div className="panel wide">
        <h3>Import Result</h3>
        <pre className="json-view">{message || 'No import run yet.'}</pre>
      </div>
    </div>
  );
}

function UploadBox({ title, accept, disabled, onFile }: { title: string; accept: string; disabled: boolean; onFile: (file?: File) => void }) {
  return (
    <label
      className="upload-box"
      onDragOver={(event) => {
        event.preventDefault();
      }}
      onDrop={(event) => {
        event.preventDefault();
        if (!disabled) onFile(event.dataTransfer.files?.[0]);
      }}
    >
      <UploadCloud size={28} />
      <strong>{title}</strong>
      <span>{accept}</span>
      <input disabled={disabled} type="file" accept={accept} onChange={(e) => onFile(e.target.files?.[0])} />
    </label>
  );
}
