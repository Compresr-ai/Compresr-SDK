import { spawn } from 'node:child_process';

export function openBrowser(url: string): boolean {
  try {
    const { command, args } = openCommand(url);
    const child = spawn(command, args, { stdio: 'ignore', detached: true });
    child.on('error', () => {
      /* headless env — caller falls back to printing the URL */
    });
    child.unref();
    return true;
  } catch {
    return false;
  }
}

function openCommand(url: string): { command: string; args: string[] } {
  switch (process.platform) {
    case 'darwin':
      return { command: 'open', args: [url] };
    case 'win32':
      // rundll32 avoids cmd.exe's `&`/`|` metacharacter splitting on URLs.
      return {
        command: 'rundll32',
        args: ['url.dll,FileProtocolHandler', url],
      };
    default:
      return { command: 'xdg-open', args: [url] };
  }
}
