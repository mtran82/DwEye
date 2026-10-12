import { useEffect, useRef, useState } from 'react'

// Blinking GESTURE_BLINKS times in a row is a hands-free command (see App).
// Each blink must follow the previous within MAX_GAP_MS; people blink on
// their own every few seconds, so five this close together is deliberate.
// A closure longer than MAX_BLINK_MS is closing the eyes, not blinking
export const GESTURE_BLINKS = 5
const MAX_GAP_MS = 1200
const MAX_BLINK_MS = 800

/**
 * Calls `onGesture` when the user blinks GESTURE_BLINKS times in a row, from
 * the server's per-frame `blink` flags. Returns how many blinks in a row so
 * far, for feedback
 */
export function useBlinkGesture(subscribe, onGesture) {
  const [count, setCount] = useState(0)
  const onGestureRef = useRef(onGesture)
  useEffect(() => {
    onGestureRef.current = onGesture
  }, [onGesture])

  useEffect(() => {
    let closedAt = null
    let lastAt = -Infinity
    let blinks = 0
    let timer

    const reset = () => {
      blinks = 0
      setCount(0)
    }
    const unsubscribe = subscribe(({ face, blink }) => {
      const now = performance.now()
      if (!face) {
        closedAt = null
        if (blinks) reset()
        return
      }
      if (blink) {
        closedAt ??= now
        return
      }
      if (closedAt === null) return
      // Eyes open again: that was a blink if it was short
      const length = now - closedAt
      closedAt = null
      if (length > MAX_BLINK_MS) {
        if (blinks) reset()
        return
      }
      blinks = now - lastAt <= MAX_GAP_MS ? blinks + 1 : 1
      lastAt = now
      clearTimeout(timer)
      if (blinks >= GESTURE_BLINKS) {
        reset()
        onGestureRef.current()
        return
      }
      setCount(blinks)
      timer = setTimeout(reset, MAX_GAP_MS)
    })
    return () => {
      unsubscribe()
      clearTimeout(timer)
    }
  }, [subscribe])

  return count
}
