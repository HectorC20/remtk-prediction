/** Logger compacto con timestamp, equivalente al log estándar del server Go. */

export function log(tag: string, ...args: unknown[]): void {
  console.log(new Date().toISOString(), `[${tag}]`, ...args);
}

export function warn(tag: string, ...args: unknown[]): void {
  console.warn(new Date().toISOString(), `[${tag}]`, ...args);
}
