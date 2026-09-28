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

// Runs server/eyetrax_server.py alongside `npm run dev` and stops it with it
function eyetraxServer() {
  return {
    name: 'eyetrax-server',
    apply: 'serve',
    configureServer(server) {
      if (!existsSync(PYTHON)) {
        server.config.logger.warn(
          `[eyetrax] ${PYTHON} not found, so the eye tracker was not started. See server/README.md`,
        )
        return
      }
      // stdin is a pipe so the server can exit if this process dies
      // without cleaning up (see --exit-with-parent)
      const child = spawn(PYTHON, ['-u', 'eyetrax_server.py', '--exit-with-parent'], {
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
