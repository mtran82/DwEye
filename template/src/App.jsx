import { useEffect, useRef } from 'react'
import './App.css'
import CalibrationTarget from './CalibrationTarget.jsx'
import { NINE_POINT_ORDER, useCalibration } from './calibration.js'
import Keyboard from './keyboard/Keyboard.jsx'
import { EYETRAX_URL, useEyetrax } from './eyetrax.js'

// Row-major cells of the 3x3 grid
const CELLS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
const RUNNING = new Set(['face', 'points', 'training'])

function targetStatus(cell, calibration) {
  const { phase } = calibration
  if (phase === 'points' && cell === calibration.cell) return calibration.stage
  if (calibration.done?.includes(cell)) return 'done'
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
    case 'training':
      return { title: 'Training…' }
    case 'done':
      return {
        title: 'Calibration complete',
        detail: `${calibration.result.samples} samples from ${calibration.result.points} points · R to recalibrate`,
      }
    case 'failed':
      return {
        title: 'Calibration failed',
        detail: `${calibration.error} · R to try again`,
        tone: 'warn',
      }
    case 'cancelled':
      return { title: 'Calibration cancelled', detail: 'R to start again' }
  }
}

function App() {
  const tracker = useEyetrax()
  const targetRefs = useRef([])
  const calibration = useCalibration(tracker, targetRefs)
  const { phase, cancel, restart } = calibration
  const { title, detail, tone } = caption(calibration, tracker.frame)

  // Esc cancels, like eyetrax's OpenCV window; R (re)starts
  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape' && RUNNING.has(phase)) cancel()
      else if (event.key === 'r' || event.key === 'R') restart()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [phase, cancel, restart])

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

      <div className="calibration-caption" data-tone={tone} aria-live="polite">
        <strong>{title}</strong>
        {detail && <span>{detail}</span>}
      </div>

      {phase === 'done' && <Keyboard frame={tracker.frame} />}
    </div>
  )
}

export default App
