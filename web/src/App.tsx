import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';
import { AccessibilityProvider } from './contexts/AccessibilityContext';
import { Layout } from './components/Layout';
import { HomePage } from './pages/HomePage';
import { ReasoningPage } from './pages/ReasoningPage';
import { ModelsPage } from './pages/ModelsPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { SettingsPage } from './pages/SettingsPage';
import { LoginPage } from './pages/LoginPage';
import { ProtectedRoute } from './components/ProtectedRoute';
import { SkipLink } from './components/accessibility/SkipLink';
import { LiveRegion } from './components/accessibility/LiveRegion';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AccessibilityProvider>
          <AuthProvider>
            <Router>
              <div className="App">
                {/* Skip navigation link for screen readers */}
                <SkipLink href="#main-content">Skip to main content</SkipLink>
                
                {/* Live region for dynamic announcements */}
                <LiveRegion />
                
                <Routes>
                  <Route path="/login" element={<LoginPage />} />
                  <Route 
                    path="/*" 
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <Routes>
                            <Route path="/" element={<HomePage />} />
                            <Route path="/reasoning" element={<ReasoningPage />} />
                            <Route path="/models" element={<ModelsPage />} />
                            <Route path="/analytics" element={<AnalyticsPage />} />
                            <Route path="/settings" element={<SettingsPage />} />
                          </Routes>
                        </Layout>
                      </ProtectedRoute>
                    } 
                  />
                </Routes>
                
                {/* Toast notifications */}
                <Toaster
                  position="top-right"
                  toastOptions={{
                    duration: 4000,
                    style: {
                      background: 'var(--toast-bg)',
                      color: 'var(--toast-text)',
                    },
                    success: {
                      iconTheme: {
                        primary: 'var(--success-color)',
                        secondary: 'white',
                      },
                    },
                    error: {
                      iconTheme: {
                        primary: 'var(--error-color)',
                        secondary: 'white',
                      },
                    },
                  }}
                />
              </div>
            </Router>
          </AuthProvider>
        </AccessibilityProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;