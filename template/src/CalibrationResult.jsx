import { useCallback, useRef } from 'react'
import { GESTURE_BLINKS } from './blinkGesture.js'
import { TARGET_CONFIDENCE } from './calibration.js'
import { useDwell } from './keyboard/useDwell.js'

// Longer than a key press, so reading the result doesn't pick a choice
const CHOICE_DWELL_MS = 1500

// By hit_rate: the share of gaze samples that landed near targets the model
// had not been trained on (see `accuracy` in server/eyetrax_server.py). High
// is what calibration aims for before it ends
const LEVELS = [
  { min: TARGET_CONFIDENCE, id: 'high', label: 'High' },
  { min: 0.5, id: 'medium', label: 'Medium' },
  { min: 0, id: 'low', label: 'Low' },
]

function Choice({ label, detail, choiceRef, active, progress }) {
  return (
    <div
      ref={choiceRef}
      className="result-choice"
      data-active={active || undefined}
    >
      <strong>{label}</strong>
      <span>{detail}</span>
      {active && (
        <span className="result-fill" style={{ transform: `scaleX(${progress})` }} />
      )}
    </div>
  )
}

/**
 * Shown after each calibration or adaptive round: tracking confidence, and
 * gaze-dwell choices to add more eyetrax adaptive points or start typing.
 */
function CalibrationResult({ result, frame, onAddPoints, onContinue }) {
  const choiceRefs = useRef({})
  const { activeKey, progress } = useDwell(frame, choiceRefs, {
    dwellMs: CHOICE_DWELL_MS,
    onCommit: (id) => (id === 'add' ? onAddPoints() : onContinue()),
  })
  // choiceRefs is filled here, not by Choice, as Keyboard does with its keys
  const registerChoice = useCallback(
    (id) => (el) => {
      choiceRefs.current[id] = el
    },
    [],
  )
  const { accuracy, points, next_round: nextRound } = result
  const level = accuracy && LEVELS.find((l) => accuracy.hit_rate >= l.min)

  return (
    <section className="result" data-level={level?.id} aria-live="polite">
      <h2>Calibration complete</h2>
      <span>Tracking confidence</span>
      {accuracy ? (
        <>
          <strong className="result-level">
            {level.label} · {Math.round(accuracy.hit_rate * 100)}%
          </strong>
          <span>
            of gaze landed within {Math.round(accuracy.tolerance_px)} px of{' '}
            {accuracy.points} targets the model had not seen · average error{' '}
            {Math.round(accuracy.error_px)} px
          </span>
          {accuracy.hit_rate <= TARGET_CONFIDENCE && !nextRound && (
            <span className="result-warn">
              Still below {Math.round(TARGET_CONFIDENCE * 100)}% with all extra points. Sitting
              still during calibration helps; blink {GESTURE_BLINKS} times to re-center, or
              recalibrate
            </span>
          )}
        </>
      ) : (
        <>
          <strong className="result-level">Not measured</strong>
          <span>Add points to measure it</span>
        </>
      )}

      <div className="result-choices">
        {nextRound > 0 ? (
          <Choice
            label={`Add ${nextRound} points`}
            detail="Improves accuracy"
            choiceRef={registerChoice('add')}
            active={activeKey === 'add'}
            progress={progress}
          />
        ) : (
          <div className="result-choice" data-disabled>
            <strong>All extra points used</strong>
          </div>
        )}
        <Choice
          label="Start typing"
          detail={`Calibrated on ${points} points`}
          choiceRef={registerChoice('type')}
          active={activeKey === 'type'}
          progress={progress}
        />
      </div>

      <small>
        Look at a choice to pick it · Any time after this, blink {GESTURE_BLINKS} times quickly
        to re-center
      </small>
    </section>
  )
}

export default CalibrationResult
