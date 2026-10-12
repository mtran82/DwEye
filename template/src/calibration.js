import { useCallback, useEffect, useRef, useState } from 'react'
import { centerOf, sleep, untilAborted } from './flow.js'

// Timings and point order follow eyetrax's calibration package
// (calibration/common.py and calibration/nine_point.py)
export const FACE_HOLD_MS = 2000 // wait_for_face_and_countdown(dur=2)
export const PULSE_MS = 1000 // _pulse_and_capture(pulse_d=1.0)
export const CAPTURE_MS = 1000 // _pulse_and_capture(cd_d=1.0)

// eyetrax's nine-point order as row-major grid cells: centre, corners, edges
export const NINE_POINT_ORDER = [4, 0, 6, 2, 8, 3, 1, 7, 5]

// Adaptive rounds (eyetrax calibration/adaptive.py) to run after the grid by
// default; each adds the server's --retrain-every points (10) and retrains.
// More can be added from the result screen
const DEFAULT_ADAPTIVE_ROUNDS = 2

// After those, calibration only ends once confidence is above this: the share
// of gaze samples landing near a round's targets before the model trained on
// them (the server's `accuracy`). Below it, another round is added, until the
// server's --random-points (60) are used up
export const TARGET_CONFIDENCE = 0.8
// How long to say why more points are coming
const MORE_POINTS_NOTICE_MS = 2500

// Re-centering after the head has moved: these targets, as fractions of the
// screen, captured from where the head is now (see RECENTER_WEIGHT in the
// server), centre first
const RECENTER_POINTS = [
  [0.5, 0.5],
  [0.2, 0.2],
  [0.8, 0.2],
  [0.8, 0.8],
  [0.2, 0.8],
]

// A point is repeated when blinks or looking away left too few samples
const MIN_SAMPLES_PER_POINT = 10
const MAX_ATTEMPTS = 3

const RESIZE_DEBOUNCE_MS = 300

/**
 * Runs eyetrax's nine-point calibration against the target elements in
 * `targetRefs` (indexed by grid cell), then adaptive rounds whose single
 * moving target is `adaptiveRef`. Starts as soon as the eye tracker is
 * connected, and starts over if the window is resized or the connection drops.
 *
 * Returns { phase, result, ...details, restart, cancel, addPoints, recenter },
 * where phase is one of
 *   connecting | face | points | adaptive | more (adding a round: confidence
 *   too low) | recenter | training | done | failed | cancelled | taken
 *   (another window started calibrating)
 * and `result` is the latest successful calibration_done from the server.
 * When done, `completed` is the kind of run that just finished ('full',
 * 'adaptive' or 'recenter') and `completedAt` when.
 */
export function useCalibration(
  { connected, connection, takeovers, send, request, subscribe },
  targetRefs,
  adaptiveRef,
) {
  // kind: 'full' calibration, one 'adaptive' round, a 'recenter', or null
  // when cancelled.
  // takeovers: as counted when the run began; more means another window took
  // over calibrating since
  const [run, setRun] = useState({ id: 0, kind: 'full', takeovers: 0 })
  const taken = takeovers > run.takeovers
  const takeoversRef = useRef(takeovers)
  const takenRef = useRef(taken)
  useEffect(() => {
    takeoversRef.current = takeovers
    takenRef.current = taken
  }, [takeovers, taken])

  // A result only holds for the connection it was made on: a reconnected
  // server may have restarted without a model
  const [saved, setSaved] = useState(null)
  const result = saved?.connection === connection ? saved.result : null
  const resultRef = useRef(result)
  useEffect(() => {
    resultRef.current = result
  }, [result])

  // Progress of the current run on the current connection. Anything else is
  // stale, e.g. 'done' from before a restart, until the run effect updates it
  const runKey = `${run.id}/${connection}`
  const [progress, setProgress] = useState(null)
  // Adding points and re-centering refine a model; without one, start over
  const refine = (run.kind === 'adaptive' || run.kind === 'recenter') && result
  const state =
    progress?.runKey === runKey
      ? progress
      : refine
        ? { phase: run.kind, point: null }
        : { phase: 'face', countingSince: null }

  useEffect(() => {
    if (!connected || !run.kind || taken) return
    const controller = new AbortController()
    const { signal } = controller
    const tracker = { send, request, subscribe }
    const show = (next) => setProgress({ ...next, runKey })
    const save = (latest) => setSaved({ connection, result: latest })

    const kind = resultRef.current ? run.kind : 'full'
    const job =
      kind === 'adaptive'
        ? adaptiveRound(tracker, adaptiveRef, show, signal)
        : kind === 'recenter'
          ? recenterRound(tracker, adaptiveRef, show, signal)
          : calibrate(tracker, targetRefs, adaptiveRef, show, save, signal)
    job.then(
      (latest) => {
        if (latest) save(latest)
        show({ phase: 'done', completed: kind, completedAt: performance.now() })
      },
      (error) => {
        if (!signal.aborted) show({ phase: 'failed', error: error.message })
      },
    )
    return () => {
      controller.abort()
      send({ type: 'calibration_cancel' })
    }
  }, [connected, connection, run, runKey, taken, send, request, subscribe, targetRefs, adaptiveRef])

  const restart = useCallback(() => {
    setSaved(null)
    setRun((r) => ({ id: r.id + 1, kind: 'full', takeovers: takeoversRef.current }))
  }, [])

  // Targets move when the window is resized (e.g. entering full screen),
  // which invalidates the calibration
  useEffect(() => {
    let timer
    let size = `${window.innerWidth}x${window.innerHeight}`
    const onResize = () => {
      clearTimeout(timer)
      timer = setTimeout(() => {
        // Resize events also fire without a size change
        const next = `${window.innerWidth}x${window.innerHeight}`
        if (next === size) return
        size = next
        // Don't take calibrating back from another window unasked
        if (takenRef.current) return
        if (resultRef.current) restart()
        else setRun((r) => (r.kind ? { ...r, id: r.id + 1, kind: 'full' } : r))
      }, RESIZE_DEBOUNCE_MS)
    }
    window.addEventListener('resize', onResize)
    return () => {
      clearTimeout(timer)
      window.removeEventListener('resize', onResize)
    }
  }, [restart])

  const addPoints = useCallback(
    () => setRun((r) => ({ ...r, id: r.id + 1, kind: 'adaptive' })),
    [],
  )
  const recenter = useCallback(
    () => setRun((r) => ({ ...r, id: r.id + 1, kind: 'recenter' })),
    [],
  )
  // Cancelling adding points or re-centering keeps the model it started from
  const cancel = useCallback(() => setRun((r) => ({ ...r, kind: null })), [])

  const actions = { result, restart, cancel, addPoints, recenter }
  if (!connected) return { phase: 'connecting', ...actions }
  if (taken) return { phase: 'taken', ...actions }
  if (!run.kind) return { phase: result ? 'done' : 'cancelled', ...actions }
  return { ...state, ...actions }
}

async function calibrate(tracker, targetRefs, adaptiveRef, show, save, signal) {
  show({ phase: 'face', countingSince: null })
  await waitForFace(tracker.subscribe, signal, (countingSince) =>
    show({ phase: 'face', countingSince }),
  )

  tracker.send({
    type: 'calibration_start',
    width: window.innerWidth,
    height: window.innerHeight,
  })
  const done = []
  for (const [step, cell] of NINE_POINT_ORDER.entries()) {
    await capturePoint(
      tracker,
      () => centerOf(targetRefs.current[cell]),
      (stage, attempt) =>
        show({ phase: 'points', step, cell, stage, attempt, done: [...done] }),
      signal,
    )
    done.push(cell)
  }

  show({ phase: 'training' })
  let latest = await finish(tracker, signal)
  // Usable from here on, even if the extra points are cancelled
  save(latest)
  for (let round = 1; ; round++) {
    const next = await adaptiveRound(tracker, adaptiveRef, show, signal)
    if (!next) break // eyetrax's point budget is used up
    latest = next
    save(latest)
    if (round < DEFAULT_ADAPTIVE_ROUNDS) continue
    const confidence = latest.accuracy?.hit_rate
    if (confidence == null || confidence > TARGET_CONFIDENCE || !latest.next_round) break
    show({ phase: 'more', confidence })
    await sleep(MORE_POINTS_NOTICE_MS, signal)
  }
  return latest
}

// One round of run_adaptive_calibration: blue-noise targets from the server,
// then retrain. Resolves to null if eyetrax's point budget is used up
async function adaptiveRound(tracker, adaptiveRef, show, signal) {
  show({ phase: 'adaptive', point: null })
  const { points } = await untilAborted(
    tracker.request({ type: 'next_points' }, 'next_points'),
    signal,
  )
  if (!points.length) return null
  for (const [step, point] of points.entries()) {
    await capturePoint(
      tracker,
      () => centerOf(adaptiveRef.current),
      (stage, attempt) =>
        show({ phase: 'adaptive', step, total: points.length, point, stage, attempt }),
      signal,
    )
  }
  show({ phase: 'training' })
  return finish(tracker, signal)
}

// RECENTER_POINTS from where the head is now, then retrain
async function recenterRound(tracker, adaptiveRef, show, signal) {
  const points = RECENTER_POINTS.map(([fx, fy]) => [
    fx * window.innerWidth,
    fy * window.innerHeight,
  ])
  for (const [step, point] of points.entries()) {
    await capturePoint(
      tracker,
      () => centerOf(adaptiveRef.current),
      (stage, attempt) =>
        show({ phase: 'recenter', step, total: points.length, point, stage, attempt }),
      signal,
      { recenter: true },
    )
  }
  show({ phase: 'training' })
  return finish(tracker, signal)
}

// eyetrax's _pulse_and_capture for one target; repeated if too few samples.
// `extra` goes with point_start
async function capturePoint({ send, request }, locate, show, signal, extra = {}) {
  for (let attempt = 1; ; attempt++) {
    show('pulse', attempt)
    await sleep(PULSE_MS, signal)

    const [x, y] = locate()
    send({ type: 'point_start', x, y, ...extra })
    show('capture', attempt)
    await sleep(CAPTURE_MS, signal)

    const { samples } = await untilAborted(
      request({ type: 'point_end' }, 'point_captured'),
      signal,
    )
    if (samples >= MIN_SAMPLES_PER_POINT || attempt === MAX_ATTEMPTS) return
  }
}

async function finish({ request }, signal) {
  const result = await untilAborted(
    request({ type: 'calibration_finish' }, 'calibration_done'),
    signal,
  )
  if (!result.ok) throw new Error(result.error)
  return result
}

// Like eyetrax's wait_for_face_and_countdown: the face must stay in view with
// eyes open for FACE_HOLD_MS; losing it restarts the countdown
function waitForFace(subscribe, signal, onCountdown) {
  return new Promise((resolve, reject) => {
    let since = null
    const unsubscribe = subscribe(({ face, blink }) => {
      if (!face || blink) {
        if (since !== null) onCountdown((since = null))
        return
      }
      const now = performance.now()
      if (since === null) onCountdown((since = now))
      else if (now - since >= FACE_HOLD_MS) {
        unsubscribe()
        resolve()
      }
    })
    signal.addEventListener(
      'abort',
      () => {
        unsubscribe()
        reject(signal.reason)
      },
      { once: true },
    )
  })
}
