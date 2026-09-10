import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Alert, Button, Popconfirm, Progress, Skeleton, Table, message } from 'antd';
import { ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { api } from '../api';
import { formatMoscow } from '../time';
import StatusTag from '../components/StatusTag';
import usePolling from '../usePolling';
import { FileProgress, taskName } from './TasksPage';

const needsRefresh = task => ['pending', 'processing'].includes(task.status) || (task.subtasks || []).some(s => s.status === 'processing');

export default function TaskDetailPage() {
  const { id } = useParams(), navigate = useNavigate();
  const { data: task, error, loading, refresh } = usePolling(useCallback(signal => api.getTask(id, signal), [id]), 2500, needsRefresh);
  const [selectedId, setSelectedId] = useState(null), [busy, setBusy] = useState('');
  useEffect(() => { setSelectedId(null); setBusy(''); }, [id]);
  async function cancel() {
    setBusy('cancel');
    try { await api.cancelTask(id); message.info('Задача отменена'); refresh(); }
    catch (e) { message.error(e.message); }
    finally { setBusy(''); }
  }
  async function restart() {
    setBusy('restart');
    try { const created = await api.restartTask(id); message.success(`Создана новая задача #${created.id}`); navigate(`/tasks/${created.id}`); }
    catch (e) { message.error(e.message); }
    finally { setBusy(''); }
  }
  const subtasks = task?.subtasks || [];
  const selected = subtasks.find(s => s.id === selectedId) || subtasks.find(s => ['processing', 'error'].includes(s.status)) || subtasks[0];
  const running = task?.status === 'processing' || subtasks.some(s => s.status === 'processing');
  const active = ['pending', 'processing'].includes(task?.status);
  const started = subtasks.map(s => s.started_at).filter(Boolean).sort()[0];
  const columns = [
    { title: 'Файл', dataIndex: 'filename', render: (name, row) => <button className="link filename" title={row.relative_path || name} onClick={() => setSelectedId(row.id)}>{row.relative_path || name}</button> },
    { title: 'Статус', dataIndex: 'status', render: s => <StatusTag status={s} />, width: 140 },
    { title: 'Прогресс', dataIndex: 'progress', width: 160, render: (p, row) => <Progress percent={Math.max(0, Math.min(100, Math.round(p || 0)))} size="small" status={row.status === 'error' ? 'exception' : row.status === 'processing' ? 'active' : undefined} /> },
    { title: 'Воркер', dataIndex: 'worker_id', render: v => v || '—', width: 160 },
    { title: 'Описание', dataIndex: 'error_message', render: v => <span className="error-description">{v || '—'}</span> },
  ];
  return <>
    <Link className="back-link" to="/tasks">← Все задачи</Link>
    {error && <Alert className="page-alert" type="error" showIcon message="Не удалось обновить задачу" description={error} action={<Button onClick={refresh}>Повторить</Button>} />}
    {!task ? <section className="panel details"><Skeleton active={loading} paragraph={{ rows: 4 }} title /></section> : <>
      <div className="pagehead"><div><div className="detailtop"><h1>Задача #{task.id}</h1><StatusTag status={task.status} /></div><p className="sub">{taskName(task.input_path)}</p></div><div className="actions">
        <Popconfirm title="Отменить задачу?" description="Готовые файлы сохранятся. Обрабатываемые файлы будут остановлены." onConfirm={cancel} okText="Отменить задачу" cancelText="Продолжить обработку" disabled={!active || !!busy}>
          <Button danger icon={<StopOutlined aria-hidden="true" />} disabled={!active || !!busy} loading={busy === 'cancel'}>Отменить</Button>
        </Popconfirm>
        <Button icon={<ReloadOutlined aria-hidden="true" />} disabled={running || !!busy} loading={busy === 'restart'} onClick={restart}>Перезапустить</Button>
      </div></div>
      {task.error_message && <Alert className="page-alert" type="error" message={task.error_message} showIcon />}
      <section className="panel details"><div className="details-grid">
        <div><label>Завершено файлов</label><FileProgress task={task} /></div><div><label>Создана · МСК</label><strong>{formatMoscow(task.created_at)}</strong></div><div><label>Начало · МСК</label><strong>{formatMoscow(started)}</strong></div><div><label>Окончание · МСК</label><strong>{formatMoscow(task.finished_at)}</strong></div>
      </div><div className="pathpair"><div><span className="pathlabel">Входные данные</span><div className="pathvalue">{task.input_path}</div></div><div><span className="pathlabel">Результаты</span><div className="pathvalue">{task.output_path}</div></div></div><div className="meta detail-uid">UID: {task.uid} · Перезапуск создаёт новую задачу с теми же путями.</div></section>
      <section className="panel"><div className="toolbar"><h2>Файлы задачи <span className="muted">{subtasks.length}</span></h2><span className="sub">Выберите файл, чтобы открыть журнал</span></div>
        <Table rowKey="id" rowClassName={row => row.id === selected?.id ? 'selected' : ''} columns={columns} dataSource={subtasks} scroll={{ x: 850 }} pagination={{ pageSize: 50, showSizeChanger: false, hideOnSinglePage: true }} />
      </section>
      <section className="panel logpanel"><div className="loghead"><div><h2>Журнал обработки</h2><div className="meta">{selected?.filename || 'Нет выбранного файла'}</div></div><span className="smallcaps">Автообновление</span></div><pre className="log" aria-label="Журнал обработки">{selected ? selected.log_text || 'Журнал пока пуст. Файл ожидает обработки.' : 'В этой задаче пока нет файлов.'}</pre></section>
      <p className="note">Время московское. Корректные готовые LAS при повторном запуске проверяются и пропускаются.</p>
    </>}
  </>;
}

