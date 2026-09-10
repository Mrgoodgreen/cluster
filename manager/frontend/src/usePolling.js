import { useEffect, useState } from 'react';

// Only one request at a time; discard responses after navigation/filter changes.
const always = () => true;
export default function usePolling(loader, interval = 3000, shouldPoll = always) {
  const [state, setState] = useState({ data: null, error: '', loading: true });
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let live = true, timer, controller;
    setState({ data: null, error: '', loading: true });
    async function poll() {
      controller = new AbortController();
      let received;
      try {
        const data = await loader(controller.signal);
        received = data;
        if (live) setState({ data, error: '', loading: false });
      } catch (error) {
        if (live) setState(current => ({ ...current, error: error.message || String(error), loading: false }));
      } finally {
        if (live && (!received || shouldPoll(received))) timer = setTimeout(poll, interval);
      }
    }
    poll();
    return () => { live = false; clearTimeout(timer); controller?.abort(); };
  }, [loader, interval, revision, shouldPoll]);
  return { ...state, refresh: () => setRevision(n => n + 1) };
}
