"use client";

import { useMutation } from "@tanstack/react-query";
import { PatientEvent, PredictResponse, postPredict } from "@/lib/api";

export function usePredict() {
  return useMutation<PredictResponse, Error, { patient: PatientEvent[] }>({
    mutationKey: ["predict"],
    mutationFn: (body) => postPredict(body),
  });
}
