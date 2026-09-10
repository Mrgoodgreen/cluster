import { useCallback, useEffect, useState } from 'react';
import { Alert, Button, Input, Pagination, Table } from 'antd';
import { PlusOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { formatMoscow } from '../time';
import StatusTag from '../components/StatusTag';
import usePolling from '../usePolling';

const filters = [['', 'Все задачи'], ['processing', 'В работе'], ['pending', 'В очереди'], ['completed', 'Завершённые'], ['error', 'Ошибки'], ['cancelled', 'Отменённые']];
export function taskName(path) { return (path || '').replace(/\\/g, '/').split('/').filter(Boolean).pop() || 'Обработка LAS'; }
export function FileProgress({ task }) {
  const total = task.subtask_total || 0, done = task.subtask_done || 0;
  return <div><div className="progresslabel">{done} <span>/ {total}</span></div><div className="progress"><i style={{ width: `${Math.min(100, done / Math.max(1, total) * 100)}%` }} /></div></div>;
}

export default function TasksPage() {
  const [page, setPage] = useState(1), [status, setStatus] = useState('');
  const [search, setSearch] = useState(''), [query, setQuery] = useState('');
  useEffect(() => { const t = setTimeout(() => { setQuery(search.trim()); setPage(1); }, 300); return () => clearTimeout(t); }, [search]);
  const { data, loading, error, refresh } = usePolling(useCallback(signal => api.listTasks(page, 20, { status, q: query }, signal), [page, status, query]));
  useEffect(() => { if (data && page > 1 && !data.items.length) setPage(Math.max(1, Math.ceil(data.total / 20))); }, [data, page]);
  const counts = data?.status_counts;
  const columns = [
    { title: 'Задача', render: (_, row) => <><Link className="taskname" title={row.input_path} to={`/tasks/${row.id}`}>{taskName(row.input_path)}</Link><div className="meta">#{row.id} · {row.subtask_total} LAS</div></> },
    { title: 'Статус', dataIndex: 'status', render: status => <StatusTag status={status} /> },
    { title: 'Завершено / всего', render: (_, row) => <FileProgress task={row} /> },
    { title: 'Создана · МСК', dataIndex: 'created_at', render: formatMoscow },
    { title: 'Окончание · МСК', dataIndex: 'finished_at', render: formatMoscow },
    { title: '', render: (_, row) => <Link to={`/tasks/${row.id}`}>Подробнее →</Link> },
  ];
  return <>
    <div className="pagehead"><div><div className="eyebrow">Обработка данных</div><h1>Задачи</h1><p className="sub">Очередь обработки и результаты классификации</p></div><Link to="/"><Button type="primary" icon={<PlusOutlined aria-hidden="true" />}>Новая задача</Button></Link></div>
    {error && <Alert className="page-alert" type="error" showIcon message="Не удалось обновить задачи" description={error} action={<Button onClick={refresh}>Повторить</Button>} />}
    <section className="panel stats" aria-label="Сводка всех задач">
      {[['processing', 'В работе'], ['pending', 'В очереди'], ['completed', 'Завершено'], ['error', 'Требуют внимания']].map(([key, label]) => <div className="stat" key={key}><label>{label}</label><strong>{counts ? counts[key] || 0 : '—'}</strong><small>задач</small></div>)}
    </section>
    <section className="panel">
      <div className="toolbar"><div className="filters" aria-label="Фильтр статуса">{filters.map(([key, label]) => <button key={key} type="button" className={`filter ${key === status ? 'active' : ''}`} aria-pressed={key === status} onClick={() => { setStatus(key); setPage(1); }}>{label}</button>)}</div><div className="toolbar-search"><Input aria-label="Поиск задачи" prefix={<SearchOutlined aria-hidden="true" />} placeholder="Путь, UID или ID" maxLength={200} allowClear value={search} onChange={e => setSearch(e.target.value)} /><Button icon={<ReloadOutlined aria-hidden="true" />} aria-label="Обновить задачи" onClick={refresh} /></div></div>
      <Table rowKey="id" loading={loading} columns={columns} dataSource={data?.items || []} pagination={false} scroll={{ x: 900 }} locale={{ emptyText: error ? 'Данные недоступны' : 'По этому запросу задач нет' }} />
      <div className="footerrow"><span>Найдено задач: {data?.total ?? '—'}</span><Pagination current={page} pageSize={20} total={data?.total || 0} onChange={setPage} showSizeChanger={false} /></div>
    </section>
    <p className="note">Сводка показывает все задачи. Список обновляется автоматически; завершённые файлы включают ошибки и отмену — подробности доступны в задаче.</p>
  </>;
}

