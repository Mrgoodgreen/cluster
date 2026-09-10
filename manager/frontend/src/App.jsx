import { useCallback } from 'react';
import { AppstoreOutlined, PlusOutlined, DeploymentUnitOutlined } from '@ant-design/icons';
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import CreateTaskPage from './pages/CreateTaskPage';
import TasksPage from './pages/TasksPage';
import TaskDetailPage from './pages/TaskDetailPage';
import { api } from './api';
import usePolling from './usePolling';

export default function App() {
  const location = useLocation();
  const health = usePolling(useCallback(signal => api.health(signal), []), 30000);
  const crumb = location.pathname === '/tasks' ? 'Задачи' : location.pathname.startsWith('/tasks/') ? `Задача #${location.pathname.split('/').pop()}` : 'Новая задача';
  return <>
    <a className="skip-link" href="#main">К содержимому</a>
    <aside className="sidebar">
      <div className="brand"><DeploymentUnitOutlined className="brand-mark" /><div><strong>TLS Classify</strong><small>Point cloud processing</small></div></div>
      <div className="navlabel">Рабочее пространство</div>
      <nav aria-label="Основная навигация">
        <NavLink to="/tasks" className={({ isActive }) => `navbtn ${isActive ? 'active' : ''}`}><AppstoreOutlined aria-hidden="true" />Задачи</NavLink>
        <NavLink to="/" end className={({ isActive }) => `navbtn ${isActive ? 'active' : ''}`}><PlusOutlined aria-hidden="true" />Создать задачу</NavLink>
      </nav>
      <div className="sidebottom"><b>TLS Cluster</b>Классификация облаков точек<br />Общее хранилище · очередь LAS</div>
    </aside>
    <div className="shell">
      <header className="topbar"><div className="breadcrumb">Рабочее пространство <span>/</span> {crumb}</div><div className="topright"><span className={`connection ${health.error ? 'offline' : health.data?.ok ? 'online' : ''}`} role="status">{health.error ? 'Нет связи с менеджером' : health.data?.ok ? 'Менеджер доступен' : 'Подключение…'}</span><span>Время · МСК</span></div></header>
      <main id="main"><Routes>
        <Route path="/" element={<CreateTaskPage />} />
        <Route path="/tasks" element={<TasksPage />} />
        <Route path="/tasks/:id" element={<TaskDetailPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes></main>
    </div>
  </>;
}

