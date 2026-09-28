import { useCallback, useState } from 'react'

/**
 * Local stand-in for fuzzykb's controller.py state model, so the fuzzykb
 * WebSocket server can later replace this hook with the same interface.
 *
 * State mirrors Controller:
 *   words      committed display words (may carry sentence punctuation)
 *   fragment   in-progress word (typed verbatim this phase)
 *   suggestions empty until the fuzzykb server exists
 */
export function useTextState() {
  const [state, setState] = useState({ words: [], fragment: '' })

  const op = useCallback((type, ch) => {
    setState((s) => {
      const words = [...s.words]
      let fragment = s.fragment
      switch (type) {
        case 'char':
          fragment += ch
          break
        case 'space':
          if (fragment) {
            words.push(fragment)
            fragment = ''
          }
          break
        case 'period':
          words.push(fragment + '.')
          fragment = ''
          break
        case 'backspace':
          if (fragment) fragment = fragment.slice(0, -1)
          else if (words.length) {
            // crossing a word boundary reopens that word for editing
            const word = words.pop()
            fragment = word.replace(/\.$/, '')
          }
          break
        default:
      }
      return { words, fragment }
    })
  }, [])

  const { words, fragment } = state
  const display =
    words.join(' ') +
    (fragment ? (words.length ? ' ' : '') + fragment : '') +
    (words.length && !fragment ? ' |' : '')

  return {
    words,
    fragment,
    display,
    suggestions: [], // fuzzykb slice wires these up
    op,
  }
}
