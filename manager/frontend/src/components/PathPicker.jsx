import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Empty, Input, Modal, Spin, message } from 'antd';
import { ArrowLeftOutlined, FolderOutlined, PlusOutlined, RightOutlined } from '@ant-design/icons';
import { api } from '../api';

export default function PathPicker({ id, value, onChange, placeholder = 'Выберите папку', disabled = false, reloadToken = 0 }) {
  const [open, setOpen] = useState(false), [current, setCurrent] = useState(null);
  const [nodes, setNodes] = useState([]), [history, setHistory] = useState([]);
  const [path, setPath] = useState(''), [name, setName] = useState('');
  const [newFolder, setNewFolder] = useState(false), [loading, setLoading] = useState(false), [saving, setSaving] = useState(false), [error, setError] = useState('');
  const request = useRef(0);
  useEffect(() => () => { request.current += 1; }, []);
  useEffect(() => { if (open) browse(current); }, [reloadToken]); // Refresh after an external mkdir.
  async function browse(destination, nextHistory = history) {
    const seq = ++request.current;
    setLoading(true); setError('');
    try {
      const result = await api.storageTree(destination);
      if (seq !== request.current) return;
      setNodes(result.nodes || []); setCurrent(destination); setPath(destination || ''); setHistory(nextHistory);
    } catch (e) { if (seq === request.current) setError(e.message); }
    finally { if (seq === request.current) setLoading(false); }
  }
  function close() { request.current += 1; setOpen(false); setSaving(false); }
  async function mkdir() {
    const trimmed = name.trim();
    if (!current || !trimmed || /[\\/]/.test(trimmed) || ['.', '..'].includes(trimmed)) { setError('Введите имя папки без разделителей пути'); return; }
    setSaving(true); setError('');
    const seq = request.current;
    try {
      const result = await api.storageMkdir(`${current.replace(/\/+$/, '')}/${trimmed}`);
      if (seq !== request.current) return;
      setName(''); setNewFolder(false); await browse(current);
      message.success(result.created ? 'Папка создана' : 'Папка уже существует');
    } catch (e) { if (seq === request.current) setError(e.message); }
    finally { if (seq === request.current || seq + 1 === request.current) setSaving(false); }
  }
  return <div className="path-picker">
    <div className="pathrow"><Input id={id} className="pathinput" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} disabled={disabled} /><Button icon={<FolderOutlined aria-hidden="true" />} disabled={disabled} onClick={() => { setOpen(true); setName(''); setNewFolder(false); browse(null, []); }}>Выбрать папку</Button></div>
    <Modal className="folder-modal" title="Выберите папку" width={670} open={open} onCancel={close} footer={<div className="modalfoot"><Button icon={<PlusOutlined aria-hidden="true" />} disabled={!current || loading || saving} onClick={() => setNewFolder(v => !v)}>Новая папка</Button><div className="actions"><Button onClick={close}>Закрыть</Button><Button type="primary" disabled={!current || loading || saving || !!error} onClick={() => { onChange(current); close(); }}>Выбрать эту папку</Button></div></div>}>
      <div className="folderbar"><Button icon={<ArrowLeftOutlined aria-hidden="true" />} aria-label="Назад по папкам" disabled={!history.length || loading || saving} onClick={() => browse(history[history.length - 1], history.slice(0, -1))} /><Input aria-label="Путь в хранилище" value={path} placeholder="Подключённые хранилища" onChange={e => setPath(e.target.value)} onPressEnter={e => { e.preventDefault(); if (!loading && !saving) browse(path.trim() || null, [...history, current]); }} /><Button disabled={loading || saving} onClick={() => browse(path.trim() || null, [...history, current])}>Перейти</Button></div>
      {error && <Alert className="page-alert" type="error" showIcon message={error} />}
      {newFolder && <div className="folderbar"><Input aria-label="Имя новой папки" autoFocus value={name} onChange={e => setName(e.target.value)} onPressEnter={e => { e.preventDefault(); if (!saving && !loading) mkdir(); }} placeholder="Имя новой папки" /><Button type="primary" loading={saving} disabled={loading} onClick={mkdir}>Создать папку</Button></div>}
      <Spin spinning={loading}><div className="folderlist">{nodes.map(node => <button type="button" className="folder" key={node.path} disabled={loading || saving || node.disabled} onClick={() => browse(node.path, [...history, current])}><FolderOutlined aria-hidden="true" /><span>{node.title}</span><RightOutlined aria-hidden="true" /></button>)}{!loading && !nodes.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={current ? 'Нет вложенных папок' : 'Нет доступных хранилищ'} />}</div></Spin>
      <p className="hint">{current || 'Выберите подключённое хранилище.'}</p>
    </Modal>
  </div>;
}

