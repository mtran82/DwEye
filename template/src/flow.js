// Helpers for the calibration flow's async steps, which stop when `signal`
// aborts

export function sleep(ms, signal) {
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

export function untilAborted(promise, signal) {
  return new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true })
    promise.then(resolve, reject)
  })
}

export function centerOf(element) {
  const r = element.getBoundingClientRect()
  return [r.left + r.width / 2, r.top + r.height / 2]
}
