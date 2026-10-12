import { useCallback, useEffect, useRef, useState } from 'react'

// server/eyetrax_server.py; override with VITE_EYETRAX_URL in .env.local
export const EYETRAX_URL =
  import.meta.env.VITE_EYETRAX_URL ?? 'ws://localhost:8765'

const RECONNECT_MS = 1000
const NO_FRAME = { face: false, blink: false, gaze: null, head: null }

/**
 * Connection to the eyetrax server. Reconnects until the server is up.
 *
 * connection counts successful connections, so state tied to one server
 *            session can tell when the server may have restarted
 * takeovers  counts times another page took over calibrating
 * frame      latest camera frame: { face, blink, gaze: [x, y] | null,
 *            head: [x, y, yaw, pitch, size] | null }
 * send       fire-and-forget message
 * request    send a message and resolve with the reply of type `replyType`
 * subscribe  call `listener` with every frame; returns an unsubscribe function
 */
export function useEyetrax(url = EYETRAX_URL) {
  const [connected, setConnected] = useState(false)
  const [connection, setConnection] = useState(0)
  const [takeovers, setTakeovers] = useState(0)
  const [frame, setFrame] = useState(NO_FRAME)
  const socketRef = useRef(null)
  const listenersRef = useRef(new Set())
  const pendingRef = useRef(new Map())

  useEffect(() => {
    const pending = pendingRef.current
    const listeners = listenersRef.current
    let disposed = false
    let retry

    const rejectPending = (reason = 'Lost connection to the eye tracker') => {
      for (const { reject } of pending.values()) reject(new Error(reason))
      pending.clear()
    }

    const connect = () => {
      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.onopen = () => {
        if (disposed) return
        setConnected(true)
        setConnection((n) => n + 1)
      }
      socket.onmessage = (event) => {
        if (disposed) return
        const message = JSON.parse(event.data)
        if (message.type === 'frame') {
          setFrame(message)
          for (const listener of listeners) listener(message)
          return
        }
        // Another page is calibrating (see the server's protocol)
        if (message.type === 'calibration_taken') {
          rejectPending('Another window started calibrating')
          setTakeovers((n) => n + 1)
          return
        }
        if (message.type === 'error') {
          rejectPending(message.error)
          return
        }
        const waiter = pending.get(message.type)
        if (waiter) {
          pending.delete(message.type)
          waiter.resolve(message)
        }
      }
      socket.onclose = () => {
        if (disposed) return
        setConnected(false)
        setFrame(NO_FRAME)
        rejectPending()
        retry = setTimeout(connect, RECONNECT_MS)
      }
    }

    connect()
    return () => {
      disposed = true
      clearTimeout(retry)
      rejectPending()
      socketRef.current?.close()
    }
  }, [url])

  const send = useCallback((message) => {
    const socket = socketRef.current
    if (socket?.readyState !== WebSocket.OPEN) return false
    socket.send(JSON.stringify(message))
    return true
  }, [])

  const request = useCallback(
    (message, replyType) =>
      new Promise((resolve, reject) => {
        pendingRef.current.set(replyType, { resolve, reject })
        if (!send(message)) {
          pendingRef.current.delete(replyType)
          reject(new Error('Eye tracker is not connected'))
        }
      }),
    [send],
  )

  const subscribe = useCallback((listener) => {
    listenersRef.current.add(listener)
    return () => listenersRef.current.delete(listener)
  }, [])

  return { connected, connection, takeovers, frame, send, request, subscribe }
}
