export interface LoggerParams {
  label: string;
}

export default class Logger {
  private readonly label: string;

  constructor(params: LoggerParams) {
    this.label = params.label;
  }

  public log(message: string): void {
    console.log(this.format("INFO", message));
  }

  public warn(message: string): void {
    console.warn(this.format("WARN", message));
  }

  public error(message: string, error?: unknown): void {
    console.error(this.format("ERROR", message), error ?? "");
  }

  private format(level: string, message: string): string {
    return `${new Date().toISOString()} [${level}] [${this.label}] ${message}`;
  }
}
