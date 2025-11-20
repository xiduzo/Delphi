export type CodeItem = {
  index: number;
  code: string;
  label: string;
};

export type PatientEvent = {
  code: string;
  age_at_event: number;
};

export type PredictResponse = {
  ranked: Array<{
    index: number;
    code: string | null;
    label: string | null;
    probability: number;
  }>;
  used_patient: PatientEvent[];
  warnings: string[];
};

// Use environment variable for API URL, fallback to localhost for development
const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888";

export async function fetchCodes(): Promise<CodeItem[]> {
  const res = await fetch(`${BASE_URL}/codes`, {
    next: { revalidate: 60 * 60 * 24 },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch codes: ${res.status}`);
  }
  return res.json();
}

export async function postPredict(body: {
  patient: PatientEvent[];
  timeframe_years?: number;
}): Promise<PredictResponse> {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Failed to predict: ${res.status}`);
  }
  return res.json();
}
