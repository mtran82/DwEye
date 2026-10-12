import react from '@vitejs/plugin-react'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

const SERVER_DIR = fileURLToPath(new URL('../server/', import.meta.url))
const PYTHON = fileURLToPath(
  new URL(
    process.platform === 'win32'
      ? '../server/.venv/Scripts/python.exe'
      : '../server/.venv/bin/python',
    import.meta.url,
  ),
)

// Runs server/eyetrax_server.py alongside `npm run dev` and stops it with it.
// EYETRAX_ARGS adds server flags (e.g. --record recordings); set
// EYETRAX_SERVER=off to run the server yourself
function eyetraxServer() {
  return {
    name: 'eyetrax-server',
    apply: 'serve',
    configureServer(server) {
      if (process.env.EYETRAX_SERVER === 'off') return
      if (!existsSync(PYTHON)) {
        server.config.logger.warn(
          `[eyetrax] ${PYTHON} not found, so the eye tracker was not started. See server/README.md`,
        )
        return
      }
      // stdin is a pipe so the server can exit if this process dies
      // without cleaning up (see --exit-with-parent)
      const extra = (process.env.EYETRAX_ARGS ?? '').split(' ').filter(Boolean)
      const child = spawn(PYTHON, ['-u', 'eyetrax_server.py', '--exit-with-parent', ...extra], {
        cwd: SERVER_DIR,
        stdio: ['pipe', 'inherit', 'inherit'],
      })
      child.on('error', (error) =>
        server.config.logger.error(`[eyetrax] Failed to start: ${error.message}`),
      )
      const stop = () => child.kill()
      process.once('exit', stop)
      server.httpServer?.once('close', () => {
        process.off('exit', stop)
        stop()
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), eyetraxServer()],
})
