/**
 * Run `fn` once when the process is asked to terminate, then exit.
 * Registered for SIGINT/SIGTERM/SIGHUP/SIGQUIT and `exit`.
 */
export function onTerminate(
  fn: ((signal: string) => void) | ((signal: string) => Promise<void>),
): void {
  let called = false;

  const wrapper = async (signal: string): Promise<void> => {
    if (called) {
      return;
    }
    called = true;

    try {
      await fn(signal);
    } catch (error) {
      console.error("Error during termination:", error);
    } finally {
      process.exit(0);
    }
  };

  for (const eventType of ["SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "exit"]) {
    process.on(eventType, (signal) => wrapper(signal));
  }
}
