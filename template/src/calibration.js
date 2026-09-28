import { useCallback, useEffect, useState } from 'react'

// Timings and point order follow eyetrax's calibration package
// (calibration/common.py and calibration/nine_point.py)
export const FACE_HOLD_MS = 2000 // wait_for_face_and_countdown(dur=2)
export const PULSE_MS = 1000 // _pulse_and_capture(pulse_d=1.0)
export const CAPTURE_MS = 1000 // _pulse_and_capture(cd_d=1.0)

// eyetrax's nine-point order as row-major grid cells: centre, corners, edges
export const NINE_POINT_ORDER = [4, 0, 6, 2, 8, 3, 1, 7, 5]

// A point is repeated when blinks or looking away left too few samples
const MIN_SAMPLES_PER_POINT = 10
const MAX_ATTEMPTS = 3

const RESIZE_DEBOUNCE_MS = 300

/**
 * Runs eyetrax's nine-point calibration against the target elements in
 * `targetRefs` (indexed by grid cell). Starts as soon as the eye tracker is
 * connected, and starts over if the window is resized.
 *
 * Returns { phase, ...details, restart, cancel }, where phase is one of
 * connecting | face | points | training | done | failed | cancelled
 */
export function useCalibration({ connected, send, request, subscribe }, targetRefs) {
  const [state, setState] = useState({ phase: 'face', countingSince: null })
  const [run, setRun] = useState({ id: 0, active: true })

  useEffect(() => {
    if (!connected || !run.active) return
    const controller = new AbortController()
    const tracker = { send, request, subscribe }
    calibrate(tracker, targetRefs, setState, controller.signal).catch((error) => {
      if (!controller.signal.aborted) {
        setState({ phase: 'failed', error: error.message })
      }
    })
    return () => {
      controller.abort()
      send({ type: 'calibration_cancel' })
    }
  }, [connected, run, send, request, subscribe, targetRefs])

  // Targets move when the window is resized (e.g. entering full screen),
  // which invalidates the calibration
  useEffect(() => {
    let timer
    const onResize = () => {
      clearTimeout(timer)
      timer = setTimeout(
        () => setRun((r) => ({ ...r, id: r.id + 1 })),
        RESIZE_DEBOUNCE_MS,
      )
    }
    window.addEventListener('resize', onResize)
    return () => {
      clearTimeout(timer)
      window.removeEventListener('resize', onResize)
    }
  }, [])

  const restart = useCallback(
    () => setRun((r) => ({ id: r.id + 1, active: true })),
    [],
  )
  const cancel = useCallback(() => setRun((r) => ({ ...r, active: false })), [])

  if (!connected) return { phase: 'connecting', restart, cancel }
  if (!run.active) return { phase: 'cancelled', restart, cancel }
  return { ...state, restart, cancel }
}

async function calibrate({ send, request, subscribe }, targetRefs, setState, signal) {
  setState({ phase: 'face', countingSince: null })
  await waitForFace(subscribe, signal, (countingSince) =>
    setState({ phase: 'face', countingSince }),
  )

  send({
    type: 'calibration_start',
    width: window.innerWidth,
    height: window.innerHeight,
  })
  const done = []
  for (const [step, cell] of NINE_POINT_ORDER.entries()) {
    for (let attempt = 1; ; attempt++) {
      setState({ phase: 'points', step, cell, stage: 'pulse', attempt, done: [...done] })
      await sleep(PULSE_MS, signal)

      const [x, y] = centerOf(targetRefs.current[cell])
      send({ type: 'point_start', x, y })
      setState((s) => ({ ...s, stage: 'capture' }))
      await sleep(CAPTURE_MS, signal)

      const { samples } = await untilAborted(
        request({ type: 'point_end' }, 'point_captured'),
        signal,
      )
      if (samples >= MIN_SAMPLES_PER_POINT || attempt === MAX_ATTEMPTS) break
    }
    done.push(cell)
  }

  setState({ phase: 'training', done })
  const result = await untilAborted(
    request({ type: 'calibration_finish' }, 'calibration_done'),
    signal,
  )
  if (!result.ok) throw new Error(result.error)
  setState({ phase: 'done', done, result })
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

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(signal.reason)
      },
      { once: true },
    )
  })
}

function untilAborted(promise, signal) {
  return new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true })
    promise.then(resolve, reject)
  })
}

function centerOf(element) {
  const r = element.getBoundingClientRect()
  return [r.left + r.width / 2, r.top + r.height / 2]
}
