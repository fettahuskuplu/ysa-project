import { createContext, useContext } from 'react';

export const DashboardContext = createContext(null);

export function useDashboard() {
  const context = useContext(DashboardContext);

  if (context === null) {
    throw new Error(
      '[useDashboard] Bu hook yalnızca <DashboardProvider> içinde kullanılabilir. ' +
      'App.jsx dosyasında <DashboardProvider> ile sarmalama yapıldığından emin olun.'
    );
  }

  return context;
}