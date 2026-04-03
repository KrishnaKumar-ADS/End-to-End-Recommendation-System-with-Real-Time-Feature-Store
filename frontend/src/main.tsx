import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

// React Query client — configure caching behavior globally
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,          // data is "fresh" for 30 seconds
      gcTime: 5 * 60_000,         // keep in cache for 5 minutes
      retry: 2,                   // retry failed requests twice
      refetchOnWindowFocus: false, // don't refetch when tab regains focus
    },
    mutations: {
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);