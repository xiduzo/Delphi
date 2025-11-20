"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CodeItem, fetchCodes } from "@/lib/api";

export function useCodes() {
  const { data, isLoading, isError, error } = useQuery<CodeItem[]>({
    queryKey: ["codes"],
    queryFn: fetchCodes,
  });

  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (!data) return [] as CodeItem[];
    const q = query.trim().toLowerCase();
    if (!q) return data;
    return data.filter(
      (item) =>
        item.code.toLowerCase().includes(q) ||
        item.label.toLowerCase().includes(q)
    );
  }, [data, query]);

  return {
    codes: data ?? [],
    filteredCodes: filtered,
    isLoading,
    isError,
    error: error as Error | null,
    query,
    setQuery,
  };
}
