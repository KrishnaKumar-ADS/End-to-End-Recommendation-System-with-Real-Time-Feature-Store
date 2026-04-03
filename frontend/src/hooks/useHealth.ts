import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/api/health";
import type { HealthResponse } from "@/types/api";

export function useHealth() {
  return useQuery<HealthResponse, Error>({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 10_000,   // poll every 10 seconds
    staleTime: 5_000,          // data is fresh for 5 seconds
    retry: 3,
    retryDelay: 2_000,
  });
}