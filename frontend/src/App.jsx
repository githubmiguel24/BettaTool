import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomeView from "./views/pages/HomeView.jsx";
import LoginView from "./views/pages/LoginView.jsx";
import RegisterView from "./views/pages/RegisterView.jsx";
import DashboardView from "./views/pages/DashboardView.jsx";
import UploadView from "./views/pages/UploadView.jsx";
import HistoryView from "./views/pages/HistoryView.jsx";
import ReportView from "./views/pages/ReportView.jsx";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomeView />} />
        <Route path="/login" element={<LoginView />} />
        <Route path="/register" element={<RegisterView />} />
        <Route path="/dashboard" element={<DashboardView />} />
        <Route path="/upload" element={<UploadView />} />
        <Route path="/history" element={<HistoryView />} />
        <Route path="/report/:id" element={<ReportView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
