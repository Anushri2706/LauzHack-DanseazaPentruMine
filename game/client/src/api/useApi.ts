import { useEffect, useState } from 'react'

/** Run an async request whenever `key` changes; keeps the last good value while loading. */
export function useApi<T>(key: string | null, fetcher: () => Promise<T>) {
  const [state, setState] = useState<{ key: string | null; data?: T; error?: string; loading: boolean }>({
    key: null, loading: false,
  })
  useEffect(() => {
    if (key === null) return
    let live = true
    setState((s) => ({ ...s, loading: true, error: undefined }))
    fetcher()
      .then((data) => live && setState({ key, data, loading: false }))
      .catch((e: Error) => live && setState((s) => ({ ...s, error: e.message, loading: false })))
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return state
}
