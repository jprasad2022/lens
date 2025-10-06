'use client';

import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { useApp } from '@/contexts/AppContext';
import { FiUser, FiLogOut, FiActivity } from 'react-icons/fi';
import { appConfig } from '@/config/app.config';
import SearchBar from '@/components/search/SearchBar';


export default function Header() {
  const { user, signOut, isAuthenticated } = useAuth();
  const { health } = useApp();

  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Navigation */}
          <div className="flex items-center space-x-8">
            <Link href="/" className="text-xl font-bold text-primary-600">
              MovieLens
            </Link>
            
            <nav className="hidden md:flex space-x-6">
              <Link href="/" className="text-gray-700 hover:text-primary-600 transition-colors">
                Recommendations
              </Link>
              
              {appConfig.features.monitoring && (
                <Link href="/monitoring" className="text-gray-700 hover:text-primary-600 transition-colors">
                  Monitoring
                </Link>
              )}
              
              {appConfig.features.abTesting && (
                <Link href="/experiments" className="text-gray-700 hover:text-primary-600 transition-colors">
                  Experiments
                </Link>
              )}
              
              <Link href="/about" className="text-gray-700 hover:text-primary-600 transition-colors">
                About
              </Link>
            </nav>
          </div>

          <div className="flex-1 flex justify-center px-2 lg:ml-6 lg:justify-end">
            <SearchBar />
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-4">
            {/* Health Status */}
            <div className="flex items-center">
              <FiActivity 
                className={`w-4 h-4 mr-2 ${
                  health?.status === 'ok' ? 'text-green-600' : 'text-red-600'
                }`} 
              />
              <span className="text-sm text-gray-600">
                API: {health?.status || 'checking...'}
              </span>
            </div>

            {/* User Menu */}
            {appConfig.features.authentication && (
              <>
                {isAuthenticated ? (
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center">
                      <FiUser className="w-4 h-4 mr-2 text-gray-600" />
                      <span className="text-sm text-gray-700">
                        {user?.email || 'User'}
                      </span>
                    </div>
                    <button
                      onClick={signOut}
                      className="p-2 text-gray-600 hover:text-red-600 transition-colors"
                      title="Sign out"
                    >
                      <FiLogOut className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <Link
                    href="/auth/signin"
                    className="px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded hover:bg-primary-700 transition-colors"
                  >
                    Sign In
                  </Link>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}