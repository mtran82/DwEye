import { useCallback, useRef } from 'react'
import './keyboard.css'
import { DWELL_MS_DEFAULT, useDwell } from './useDwell.js'
import { useTextState } from './useTextState.js'

// QWERTY; three letter rows then the action row
const ROWS = [
  ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
  ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
  ['z', 'x', 'c', 'v', 'b', 'n', 'm', "'"],
]

function GazeDot({ frame }) {
  const [x, y] = frame.gaze ?? []
  return (
    <div
      className="gaze-cursor"
      data-visible={frame.gaze != null}
      style={{ transform: `translate(${x ?? 0}px, ${y ?? 0}px)` }}
    />
  )
}

function Key({ id, label, keyRef, active, progress }) {
  return (
    <div
      ref={keyRef}
      className="kb-key"
      data-active={active || undefined}
      data-action={id.length > 1 || undefined}
      aria-hidden="true"
    >
      {label}
      {active && (
        <span className="kb-fill" style={{ transform: `scale(${progress})` }} />
      )}
    </div>
  )
}

/**
 * Dwell keyboard shown after calibration completes. Gaze coordinates are CSS
 * pixels relative to the viewport, exactly what the calibration maps to.
 * Suggestions render as empty slots until the fuzzykb server is wired.
 */
export default function Keyboard({ frame, dwellMs = DWELL_MS_DEFAULT }) {
  const keyRefs = useRef({})
  const text = useTextState()

  const commit = (id) => {
    if (id === 'space') text.op('space')
    else if (id === 'period') text.op('period')
    else if (id === 'backspace') text.op('backspace')
    else text.op('char', id)
  }
  const { activeKey, progress } = useDwell(frame, keyRefs, { dwellMs, onCommit: commit })
  // Stable across renders so React doesn't detach/reattach every key's ref
  const registerKey = useCallback(
    (id) => (el) => {
      keyRefs.current[id] = el
    },
    [],
  )

  return (
    <div className="kb">
      <GazeDot frame={frame} />

      {text.suggestions.length > 0 && (
        <div className="kb-suggestions" aria-hidden="true">
          {text.suggestions.map((s, i) => (
            <div key={i} className="kb-suggestion">
              {s.word}
            </div>
          ))}
        </div>
      )}

      <div className="kb-text" aria-hidden="true">
        {text.display}
      </div>

      <div className="kb-rows">
        {ROWS.map((row, r) => (
          <div key={r} className="kb-row">
            {row.map((ch) => (
              <Key
                key={ch}
                id={ch}
                label={ch}
                keyRef={registerKey(ch)}
                active={activeKey === ch}
                progress={progress}
              />
            ))}
          </div>
        ))}
        <div className="kb-row kb-actions">
          <Key
            id="backspace"
            label="⌫"
            keyRef={registerKey('backspace')}
            active={activeKey === 'backspace'}
            progress={progress}
          />
          <Key
            id="space"
            label="space"
            keyRef={registerKey('space')}
            active={activeKey === 'space'}
            progress={progress}
          />
          <Key
            id="period"
            label="."
            keyRef={registerKey('period')}
            active={activeKey === 'period'}
            progress={progress}
          />
        </div>
      </div>
    </div>
  )
}
