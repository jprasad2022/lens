/**
 * Authentication Service (Optional)
 * Handles Firebase authentication if enabled
 */

import { appConfig } from '@/config/app.config';

class AuthService {
  constructor() {
    this.auth = null;
    this.currentUser = null;
    this.initialized = false;
  }

  async initialize() {
    if (!appConfig.features.authentication || this.initialized) {
      return;
    }

    try {
      const { initializeApp } = await import('firebase/app');
      const { getAuth, onAuthStateChanged } = await import('firebase/auth');

      const firebaseConfig = {
        apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
        authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
        projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
        storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
        messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
        appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
      };

      const app = initializeApp(firebaseConfig);
      this.auth = getAuth(app);

      // Listen to auth state changes
      onAuthStateChanged(this.auth, (user) => {
        this.currentUser = user;
      });

      this.initialized = true;
    } catch (error) {
      console.error('Failed to initialize Firebase:', error);
    }
  }

  async getIdToken() {
    if (!this.currentUser) return null;
    try {
      return await this.currentUser.getIdToken();
    } catch (error) {
      console.error('Failed to get ID token:', error);
      return null;
    }
  }

  async refreshToken() {
    if (!this.currentUser) return;
    try {
      return await this.currentUser.getIdToken(true);
    } catch (error) {
      console.error('Failed to refresh token:', error);
    }
  }

  async signIn(email, password) {
    if (!this.auth) {
      throw new Error('Authentication not initialized');
    }

    const { signInWithEmailAndPassword } = await import('firebase/auth');
    return signInWithEmailAndPassword(this.auth, email, password);
  }

  async signOut() {
    if (!this.auth) return;
    const { signOut } = await import('firebase/auth');
    return signOut(this.auth);
  }

  isAuthenticated() {
    return !!this.currentUser;
  }

  getCurrentUser() {
    return this.currentUser;
  }
}

// Export a singleton instance
export const authService = new AuthService();

// Initialize on import if authentication is enabled
if (appConfig.features.authentication && typeof window !== 'undefined') {
  authService.initialize();
}