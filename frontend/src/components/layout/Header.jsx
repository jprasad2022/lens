'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useApp } from '@/contexts/AppContext';
import { FiUser, FiLogOut, FiActivity, FiMenu, FiX, FiFilm, FiSearch } from 'react-icons/fi';
import { appConfig } from '@/config/app.config';
import SearchBar from '@/components/search/SearchBar';

export default function Header() {
  const { user, signOut, isAuthenticated } = useAuth();
  const { health } = useApp();
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className={`sticky top-0 z-50 transition-all duration-300 ${
      isScrolled 
        ? 'bg-white/95 dark:bg-gray-900/95 backdrop-blur-md shadow-lg' 
        : 'bg-white dark:bg-gray-900'
    }`}>
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Navigation */}
          <div className="flex items-center space-x-8">
            <Link 
              href="/" 
              className="flex items-center space-x-2 group"
            >
              <div className="p-2 bg-gradient-to-br from-primary-600 to-secondary-600 rounded-lg group-hover:shadow-glow transition-all duration-300">
                <FiFilm className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
                MovieLens
              </span>
            </Link>
            
            <nav className="hidden md:flex space-x-6">
              <Link 
                href="/" 
                className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors font-medium"
              >
                Recommendations
              </Link>
              
              {appConfig.features.monitoring && (
                <Link 
                  href="/monitoring" 
                  className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors font-medium"
                >
                  Monitoring
                </Link>
              )}
              
              {appConfig.features.abTesting && (
                <Link 
                  href="/experiments" 
                  className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors font-medium"
                >
                  Experiments
                </Link>
              )}
              
              <Link 
                href="/about" 
                className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors font-medium"
              >
                About
              </Link>
            </nav>
          </div>

          {/* Search Bar - Desktop */}
          <div className="hidden md:flex flex-1 max-w-xl mx-8">
            <SearchBar />
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-4">
            {/* Health Status */}
            <div className={`hidden md:flex items-center px-3 py-1.5 rounded-full ${
              health?.status === 'ok' 
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' 
                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
            }`}>
              <FiActivity className="w-4 h-4 mr-1.5" />
              <span className="text-sm font-medium">
                {health?.status || 'checking...'}
              </span>
            </div>

            {/* User Menu */}
            {appConfig.features.authentication && (
              <>
                {isAuthenticated ? (
                  <div className="hidden md:flex items-center space-x-3">
                    <div className="flex items-center px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-full">
                      <FiUser className="w-4 h-4 mr-2 text-gray-600 dark:text-gray-400" />
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {user?.email || 'User'}
                      </span>
                    </div>
                    <button
                      onClick={signOut}
                      className="p-2 text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-all duration-200"
                      title="Sign out"
                    >
                      <FiLogOut className="w-5 h-5" />
                    </button>
                  </div>
                ) : (
                  <Link
                    href="/auth/signin"
                    className="hidden md:inline-flex btn-primary text-sm"
                  >
                    Sign In
                  </Link>
                )}
              </>
            )}

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              {isMobileMenuOpen ? <FiX className="w-6 h-6" /> : <FiMenu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-gray-200 dark:border-gray-700 animate-slide-up">
            {/* Mobile Search */}
            <div className="mb-4">
              <SearchBar />
            </div>

            {/* Mobile Navigation */}
            <nav className="space-y-2 mb-4">
              <Link 
                href="/" 
                onClick={() => setIsMobileMenuOpen(false)}
                className="block px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
              >
                <FiFilm className="inline-block w-4 h-4 mr-2" />
                Recommendations
              </Link>
              
              {appConfig.features.monitoring && (
                <Link 
                  href="/monitoring" 
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="block px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
                >
                  Monitoring
                </Link>
              )}
              
              {appConfig.features.abTesting && (
                <Link 
                  href="/experiments" 
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="block px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
                >
                  Experiments
                </Link>
              )}
              
              <Link 
                href="/about" 
                onClick={() => setIsMobileMenuOpen(false)}
                className="block px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
              >
                About
              </Link>
            </nav>

            {/* Mobile Health Status */}
            <div className="px-4 py-2 mb-4">
              <div className={`inline-flex items-center px-3 py-1.5 rounded-full ${
                health?.status === 'ok' 
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' 
                  : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
              }`}>
                <FiActivity className="w-4 h-4 mr-1.5" />
                <span className="text-sm font-medium">
                  API: {health?.status || 'checking...'}
                </span>
              </div>
            </div>

            {/* Mobile Auth */}
            {appConfig.features.authentication && (
              <div className="px-4">
                {isAuthenticated ? (
                  <div className="space-y-2">
                    <div className="flex items-center px-3 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                      <FiUser className="w-4 h-4 mr-2 text-gray-600 dark:text-gray-400" />
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {user?.email || 'User'}
                      </span>
                    </div>
                    <button
                      onClick={signOut}
                      className="w-full flex items-center justify-center px-4 py-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                    >
                      <FiLogOut className="w-4 h-4 mr-2" />
                      Sign Out
                    </button>
                  </div>
                ) : (
                  <Link
                    href="/auth/signin"
                    onClick={() => setIsMobileMenuOpen(false)}
                    className="block w-full text-center btn-primary text-sm"
                  >
                    Sign In
                  </Link>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
}