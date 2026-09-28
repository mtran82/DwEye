import { CAPTURE_MS, FACE_HOLD_MS } from './calibration.js'

/**
 * One calibration point. `status` is one of:
 *   pending    faint ring, waiting its turn
 *   countdown  ring runs down while eyetrax waits for a steady face
 *   pulse      dot pulses to draw the eye to it
 *   capture    dot holds still while the ring runs down and samples are taken
 *   done       sampled; fades away
 */
function CalibrationTarget({ ref, status }) {
  const ringMs = status === 'countdown' ? FACE_HOLD_MS : CAPTURE_MS
  return (
    <div ref={ref} className="calibration-circle" data-status={status} aria-hidden="true">
      <span className="calibration-dot" />
      {(status === 'countdown' || status === 'capture') && (
        <svg
          className="calibration-ring"
          viewBox="0 0 80 80"
          style={{ '--ring-ms': `${ringMs}ms` }}
        >
          <circle cx="40" cy="40" r="37" pathLength="100" />
        </svg>
      )}
    </div>
  )
}

export default CalibrationTarget
