export async function isInternetAvailable(): Promise<boolean> {
  try {
    await fetch("https://www.google.com", { mode: "no-cors" });
    return true;
  } catch {
    return false;
  }
}
