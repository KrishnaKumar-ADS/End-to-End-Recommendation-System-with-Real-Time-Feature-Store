import { Routes, Route } from "react-router-dom";
import Home from "@/pages/Home";
import ABDashboard from "@/pages/ABDashboard";
import Navbar from "@/components/Navbar";

export default function App() {
  return (
    <div className="min-h-screen bg-cinema-bg">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/ab-dashboard" element={<ABDashboard />} />
        </Routes>
      </main>
    </div>
  );
}