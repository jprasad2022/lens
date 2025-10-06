'use client';

import { createContext, useContext, useEffect, useState } from 'react';
import { authService } from '@/services/auth.service';
import { appConfig } from '@/config/app.config';

const AuthContext = createContext({
  user: null,
  loading: true,
  signIn: async () => {},
  signOut: async () => {},
  isAuthenticated: false,
});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!appConfig.features.authentication) {
      setLoading(false);
      return;
    }

    const initAuth = async () => {
      try {
        await authService.initialize();
        const currentUser = authService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error('Auth initialization failed:', error);
      } finally {
        setLoading(false);
      }
    };

    initAuth();
  }, []);

  const signIn = async (email, password) => {
    if (!appConfig.features.authentication) {
      throw new Error('Authentication is not enabled');
    }
    const userCredential = await authService.signIn(email, password);
    setUser(userCredential.user);
    return userCredential;
  };

  const signOut = async () => {
    await authService.signOut();
    setUser(null);
  };

  const value = {
    user,
    loading,
    signIn,
    signOut,
    isAuthenticated: !!user,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};