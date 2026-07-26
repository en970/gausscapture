import { useState } from 'react';
import { Plus } from 'lucide-react';

type Props = {
  onCreate: (name: string, target: string) => Promise<void>;
};

export default function ProjectCreate({ onCreate }: Props) {
  const [name, setName] = useState('Living Room Test');
  const [target, setTarget] = useState('room');
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      await onCreate(name, target);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="create-form" onSubmit={submit}>
      <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Project name" />
      <select value={target} onChange={(e) => setTarget(e.target.value)}>
        <option value="room">Room / Interior</option>
        <option value="object">Object</option>
        <option value="outdoor">Outdoor</option>
        <option value="unknown">Unknown</option>
      </select>
      <button className="primary" disabled={busy}>
        <Plus size={18} /> New Project
      </button>
    </form>
  );
}

