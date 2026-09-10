const labels = { pending: 'В очереди', processing: 'В работе', completed: 'Завершена',
  success: 'Готово', skipped: 'Пропущен', error: 'Ошибка', cancelled: 'Отменена' };

export default function StatusTag({ status }) {
  return <span className={`status ${Object.hasOwn(labels, status) ? status : 'unknown'}`}>{labels[status] || status || '—'}</span>;
}
