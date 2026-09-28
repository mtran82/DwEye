import { useCallback, useEffect, useRef, useState } from 'react'

// server/eyetrax_server.py; override with VITE_EYETRAX_URL in .env.local
export const EYETRAX_URL =
  import.meta.env.VITE_EYETRAX_URL ?? 'ws://localhost:8765'

const RECONNECT_MS = 1000
const NO_FRAME = { face: false, blink: false, gaze: null }

/**
 * Connection to the eyetrax server. Reconnects until the server is up.
 *
 * frame      latest camera frame: { face, blink, gaze: [x, y] | null }
 * send       fire-and-forget message
 * request    send a message and resolve with the reply of type `replyType`
 * subscribe  call `listener` with every frame; returns an unsubscribe function
 */
export function useEyetrax(url = EYETRAX_URL) {
  const [connected, setConnected] = useState(false)
  const [frame, setFrame] = useState(NO_FRAME)
  const socketRef = useRef(null)
  const listenersRef = useRef(new Set())
  const pendingRef = useRef(new Map())

  useEffect(() => {
    const pending = pendingRef.current
    const listeners = listenersRef.current
    let disposed = false
    let retry

    const rejectPending = () => {
      for (const { reject } of pending.values()) {
        reject(new Error('Lost connection to the eye tracker'))
      }
      pending.clear()
    }

    const connect = () => {
      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.onopen = () => {
        if (!disposed) setConnected(true)
      }
      socket.onmessage = (event) => {
        if (disposed) return
        const message = JSON.parse(event.data)
        if (message.type === 'frame') {
          setFrame(message)
          for (const listener of listeners) listener(message)
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

  return { connected, frame, send, request, subscribe }
}
