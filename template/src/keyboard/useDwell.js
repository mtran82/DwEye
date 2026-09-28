import { useEffect, useRef, useState } from 'react'

export const DWELL_MS_DEFAULT = 800

/**
 * Gaze-dwell selection engine.
 *
 * Driven by each `frame` update (~60/s from the eyetrax server). Accumulates
 * how long the gaze has held on the same key; at `dwellMs` fires `onCommit`
 * with the key id. Looking away, losing the face, or blinking resets the
 * timer (a blink freezes the cursor but never selects).
 *
 * Returns { activeKey, progress } for the UI to highlight and fill.
 */
export function useDwell(frame, keyRefs, { dwellMs = DWELL_MS_DEFAULT, onCommit }) {
  const [activeKey, setActiveKey] = useState(null)
  const [progress, setProgress] = useState(0)

  const dwellRef = useRef({ key: null, since: 0 })
  const commitRef = useRef(onCommit)
  // Latest-ref pattern: onCommit changes are picked up without re-subscribing
  useEffect(() => {
    commitRef.current = onCommit
  }, [onCommit])

  useEffect(() => {
    const { gaze, face, blink } = frame
    const reset = () => {
      if (dwellRef.current.key !== null) {
        dwellRef.current = { key: null, since: 0 }
        setActiveKey(null)
        setProgress(0)
      }
    }
    if (!gaze || !face || blink) {
      reset()
      return
    }

    const [x, y] = gaze
    let hit = null
    for (const [id, el] of Object.entries(keyRefs.current)) {
      if (!el) continue
      const r = el.getBoundingClientRect()
      if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) {
        hit = id
        break
      }
    }

    if (hit !== dwellRef.current.key) {
      dwellRef.current = hit === null ? { key: null, since: 0 } : { key: hit, since: performance.now() }
      setActiveKey(hit)
      setProgress(0)
      return
    }

    if (hit === null) return
    const held = performance.now() - dwellRef.current.since
    // Quantised so unchanged values bail out of the re-render
    setProgress(Math.min(1, Math.floor((held / dwellMs) * 20) / 20))
    if (held >= dwellMs) {
      commitRef.current(hit)
      // Full dwell again before re-firing, but only after re-entering the key
      dwellRef.current = { key: null, since: 0 }
      setActiveKey(null)
      setProgress(0)
    }
  }, [frame, keyRefs, dwellMs])

  return { activeKey, progress }
}
