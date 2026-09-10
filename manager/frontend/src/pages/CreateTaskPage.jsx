import { useMemo, useState } from 'react';
import { Alert, Button, Form, Input, Switch, message } from 'antd';
import { ArrowRightOutlined, FolderOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import PathPicker from '../components/PathPicker';
import { api } from '../api';

const normalize = p => (p || '').trim().replace(/\\/g, '/').replace(/\/+$/, '');
export function autoClassifiedPath(inputPath) {
  const p = normalize(inputPath);
  return !p ? '' : p.endsWith('_classified') ? p : `${p}_classified`;
}
export default function CreateTaskPage() {
  const [inputPath, setInputPath] = useState(''), [outputPath, setOutputPath] = useState('');
  const [autoOutput, setAutoOutput] = useState(true), [busy, setBusy] = useState('');
  const [error, setError] = useState(''), [treeReload, setTreeReload] = useState(0);
  const navigate = useNavigate();
  const effectiveOutput = useMemo(() => autoOutput ? autoClassifiedPath(inputPath) : outputPath, [autoOutput, inputPath, outputPath]);
  async function createFolder() {
    setError('');
    if (!normalize(effectiveOutput)) { setError('Укажите путь папки для создания'); return; }
    setBusy('mkdir');
    try { const result = await api.storageMkdir(normalize(effectiveOutput)); if (!autoOutput) setOutputPath(result.path); setTreeReload(n => n + 1); message.success(result.created ? 'Папка создана' : 'Папка уже существует'); }
    catch (e) { setError(e.message); }
    finally { setBusy(''); }
  }
  async function createTask() {
    setError('');
    const input = normalize(inputPath), output = normalize(effectiveOutput);
    if (!input || !output) { setError('Укажите входную и выходную папки'); return; }
    if (input === output) { setError('Входная и выходная папки должны различаться'); return; }
    setBusy('task');
    try { await api.storageMkdir(output); const task = await api.createTask(input, output); message.success(`Задача #${task.id} создана`); navigate(`/tasks/${task.id}`); }
    catch (e) { setError(e.message); }
    finally { setBusy(''); }
  }
  return <>
    <div className="pagehead"><div><div className="eyebrow">Новая обработка</div><h1>Создать задачу</h1><p className="sub">Выберите папку с LAS и место для готовых облаков</p></div></div>
    <div className="create-grid"><section className="panel formpanel"><div className="section-title"><FolderOutlined aria-hidden="true" /><h2>Расположение данных</h2></div>
      {error && <Alert className="page-alert" type="error" showIcon message={error} />}
      <Form layout="vertical" className="create-task-form" onFinish={createTask}>
        <Form.Item label="Входные данные" htmlFor="inputPath" required><PathPicker id="inputPath" value={inputPath} onChange={setInputPath} placeholder="Путь к исходной папке" disabled={!!busy} reloadToken={treeReload} /><p className="hint">Папка на общем хранилище, доступном менеджеру и воркерам.</p></Form.Item>
        <div className="togglebox"><div><label htmlFor="autoOutput">Создать выходную папку автоматически</label><p className="hint">Рядом с исходной, с суффиксом <b>_classified</b></p></div><Switch id="autoOutput" aria-label="Автоматическая выходная папка" checked={autoOutput} disabled={!!busy} onChange={checked => { if (!checked && !outputPath) setOutputPath(effectiveOutput); setAutoOutput(checked); }} /></div>
        <Form.Item label="Результаты классификации" htmlFor="outputPath" required>
          {autoOutput ? <Input id="outputPath" className="pathinput" value={effectiveOutput} readOnly placeholder="…_classified" /> : <PathPicker id="outputPath" value={outputPath} onChange={setOutputPath} placeholder="Папка результатов" disabled={!!busy} reloadToken={treeReload} />}
          <p className="hint">{autoOutput ? 'Папка будет создана при постановке задачи.' : 'Выберите папку или введите новый путь.'}</p>
          <Button className="create-folder-link" type="link" disabled={!!busy || !effectiveOutput} loading={busy === 'mkdir'} onClick={createFolder}>+ Создать выходную папку заранее</Button>
        </Form.Item>
        <div className="formfoot"><span className="hint">Готовые результаты будут проверены и пропущены.</span><Button type="primary" htmlType="submit" disabled={!!busy} loading={busy === 'task'}>Поставить в очередь <ArrowRightOutlined aria-hidden="true" /></Button></div>
      </Form>
    </section><aside className="panel guide"><div className="eyebrow">Как это работает</div>{[['Выберите данные', 'Каждый LAS обрабатывается отдельно. Свободные воркеры забирают следующие файлы из очереди.'], ['Следите за обработкой', 'В деталях задачи доступны статус каждого файла, имя воркера и журнал выполнения.'], ['Проверьте результат', 'Готовые облака сохраняются в выходную папку с сохранением структуры.']].map(([title, text], i) => <section key={title}><h3><span className="guide-number">{i + 1}</span>{title}</h3><p>{text}</p></section>)}</aside></div>
  </>;
}

