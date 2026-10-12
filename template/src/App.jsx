import { useEffect, useRef, useState } from 'react'
import './App.css'
import CalibrationResult from './CalibrationResult.jsx'
import CalibrationTarget from './CalibrationTarget.jsx'
import { GESTURE_BLINKS, useBlinkGesture } from './blinkGesture.js'
import { NINE_POINT_ORDER, TARGET_CONFIDENCE, useCalibration } from './calibration.js'
import Keyboard from './keyboard/Keyboard.jsx'
import { EYETRAX_URL, useEyetrax } from './eyetrax.js'

// Row-major cells of the 3x3 grid
const CELLS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
const RUNNING = new Set(['face', 'points', 'adaptive', 'more', 'recenter', 'training'])
// Where blinking GESTURE_BLINKS times starts calibrating again, hands-free
const RESTARTABLE = new Set(['failed', 'cancelled', 'taken'])
// After a re-center, blinking again within this long recalibrates fully
const RECALIBRATE_WINDOW_MS = 8000

function targetStatus(cell, calibration) {
  const { phase } = calibration
  if (phase === 'points') {
    if (cell === calibration.cell) return calibration.stage
    return calibration.done.includes(cell) ? 'done' : 'pending'
  }
  if (['adaptive', 'more', 'recenter', 'training', 'done'].includes(phase)) return 'done'
  if (
    phase === 'face' &&
    calibration.countingSince != null &&
    cell === NINE_POINT_ORDER[0]
  ) {
    return 'countdown'
  }
  return 'pending'
}

function caption(calibration, frame) {
  const restartHint = `Blink ${GESTURE_BLINKS} times quickly, or press R, to`
  switch (calibration.phase) {
    case 'connecting':
      return {
        title: 'Connecting to the eye tracker…',
        detail: `Waiting for the eyetrax server at ${EYETRAX_URL}`,
      }
    case 'face':
      return frame.face
        ? { title: 'Hold still…', detail: 'Calibration is about to start' }
        : {
            title: 'Face not detected',
            detail: 'Sit in front of the camera and look at the screen',
            tone: 'warn',
          }
    case 'points':
      if (!frame.face) {
        return { title: "Can't see your eyes", detail: 'Face the camera', tone: 'warn' }
      }
      return {
        title: calibration.attempt > 1 ? 'Keep looking at the dot' : 'Look at the dot',
        detail: `${calibration.step + 1} / ${NINE_POINT_ORDER.length}`,
      }
    case 'adaptive':
    case 'recenter': {
      const { phase, point, step, total, attempt } = calibration
      const recentering = phase === 'recenter'
      if (!point) return { title: recentering ? 'Re-centering…' : 'Adding points…' }
      // Keep the caption clear of the target
      const position = point[1] > window.innerHeight * 0.55 ? 'top' : undefined
      if (!frame.face) {
        return { title: "Can't see your eyes", detail: 'Face the camera', tone: 'warn', position }
      }
      return {
        title: attempt > 1 ? 'Keep looking at the dot' : 'Look at the dot',
        detail: `${recentering ? 'Re-centering' : 'Extra point'} ${step + 1} / ${total} · Esc to skip`,
        position,
      }
    }
    case 'more':
      return {
        title: `Confidence ${Math.round(calibration.confidence * 100)}%, below ${Math.round(TARGET_CONFIDENCE * 100)}%`,
        detail: 'Adding 10 more points',
      }
    case 'training':
      return { title: 'Training…' }
    case 'done':
      // CalibrationResult, then the keyboard, fill the screen
      return null
    case 'failed':
      return {
        title: 'Calibration failed',
        detail: `${calibration.error} · ${restartHint} try again`,
        tone: 'warn',
      }
    case 'taken':
      return {
        title: 'Calibrating in another window',
        detail: `Close the other window, or ${restartHint.toLowerCase()} calibrate here instead`,
        tone: 'warn',
      }
    case 'cancelled':
      return { title: 'Calibration cancelled', detail: `${restartHint} start again` }
  }
}

// Live gaze; fades out in place during blinks
function GazeCursor({ gaze }) {
  const [position, setPosition] = useState(gaze)
  if (gaze && gaze !== position) setPosition(gaze)
  if (!position) return null

  const [x, y] = position
  return (
    <div
      className="gaze-cursor"
      data-visible={gaze != null}
      style={{ transform: `translate(${x}px, ${y}px)` }}
    />
  )
}

// Feedback while blinking the gesture
function BlinkCount({ count }) {
  return (
    <div className="notice" role="status">
      Blinks
      {Array.from({ length: GESTURE_BLINKS }, (_, i) => (
        <span key={i} className="notice-dot" data-on={i < count || undefined} />
      ))}
    </div>
  )
}

// After a re-center, while blinking again would recalibrate fully: until
// RECALIBRATE_WINDOW_MS after `since`
function RecenterNotice({ since }) {
  const until = since + RECALIBRATE_WINDOW_MS
  const [visible, setVisible] = useState(() => performance.now() < until)
  useEffect(() => {
    const timer = setTimeout(() => setVisible(false), until - performance.now())
    return () => clearTimeout(timer)
  }, [until])
  if (!visible) return null
  return (
    <div className="notice" role="status">
      Re-centered · Blink {GESTURE_BLINKS} times again now to recalibrate fully
    </div>
  )
}

function App() {
  const tracker = useEyetrax()
  const targetRefs = useRef([])
  const adaptiveRef = useRef(null)
  const calibration = useCalibration(tracker, targetRefs, adaptiveRef)
  const { phase, result, completed, completedAt, cancel, restart, addPoints, recenter } =
    calibration
  const message = caption(calibration, tracker.frame)
  const canAddPoints = phase === 'done' && result?.next_round > 0
  // The result the user moved on from; a newer one shows the result screen
  // again, except after re-centering, which goes straight back to typing
  const [accepted, setAccepted] = useState(null)
  const typing = phase === 'done' && (accepted === result || completed === 'recenter')

  // Blinking GESTURE_BLINKS times: re-center after calibrating (or, right
  // after a re-center, recalibrate fully); start again if calibration stopped
  const blinks = useBlinkGesture(tracker.subscribe, () => {
    if (phase === 'done') {
      const justRecentered =
        completed === 'recenter' && performance.now() - completedAt < RECALIBRATE_WINDOW_MS
      if (justRecentered) restart()
      else recenter()
    } else if (RESTARTABLE.has(phase)) {
      restart()
    }
  })
  const gestureOn = phase === 'done' || RESTARTABLE.has(phase)

  // Esc cancels, like eyetrax's OpenCV window; R (re)starts; A adds points;
  // Enter moves on from the result screen
  useEffect(() => {
    const onKeyDown = (event) => {
      const key = event.key.toLowerCase()
      if (key === 'escape' && RUNNING.has(phase)) cancel()
      else if (key === 'r') restart()
      else if (key === 'a' && canAddPoints) addPoints()
      else if (key === 'enter' && phase === 'done') setAccepted(result)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [phase, result, canAddPoints, cancel, restart, addPoints])

  const moving = (phase === 'adaptive' || phase === 'recenter') && calibration.point

  return (
    <div className="calibration">
      <div className="calibration-grid">
        {CELLS.map((cell) => (
          <CalibrationTarget
            key={cell}
            ref={(el) => {
              targetRefs.current[cell] = el
            }}
            status={targetStatus(cell, calibration)}
          />
        ))}
      </div>

      {moving && (
        <div
          className="adaptive-target"
          style={{ left: calibration.point[0], top: calibration.point[1] }}
        >
          <CalibrationTarget key={calibration.step} ref={adaptiveRef} status={calibration.stage} />
        </div>
      )}

      {message && (
        <div
          className="calibration-caption"
          data-tone={message.tone}
          data-position={message.position}
          aria-live="polite"
        >
          <strong>{message.title}</strong>
          {message.detail && <span>{message.detail}</span>}
        </div>
      )}

      {phase === 'done' && !typing && (
        <>
          <CalibrationResult
            result={result}
            frame={tracker.frame}
            onAddPoints={addPoints}
            onContinue={() => setAccepted(result)}
          />
          <GazeCursor gaze={tracker.frame.gaze} />
        </>
      )}
      {typing && <Keyboard frame={tracker.frame} />}

      {gestureOn && blinks >= 2 ? (
        <BlinkCount count={blinks} />
      ) : (
        phase === 'done' &&
        completed === 'recenter' && <RecenterNotice key={completedAt} since={completedAt} />
      )}
    </div>
  )
}

export default App
