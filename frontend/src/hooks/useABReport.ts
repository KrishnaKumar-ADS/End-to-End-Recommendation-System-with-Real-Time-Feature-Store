import { useQuery } from "@tanstack/react-query";
import { fetchABReport } from "@/api/recommendations";
import type { ABReport } from "@/types/api";

export function useABReport() {
  return useQuery<ABReport, Error>({
    queryKey: ["ab-report"],
    queryFn: fetchABReport,
    refetchInterval: 30_000,   // refresh every 30 seconds
    staleTime: 15_000,
  });
}