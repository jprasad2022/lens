import './globals.css';
import { Inter } from 'next/font/google';
import Providers from './providers';
import Header from '@/components/layout/Header';
import { appConfig } from '@/config/app.config';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'MovieLens Recommender',
  description: 'Cloud-native movie recommendation system with real-time ML capabilities',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.className}>
      <body className="bg-gray-50">
        <Providers>
          <div className="min-h-screen flex flex-col">
            <Header />
            <main className="flex-1">
              {children}
            </main>
            {appConfig.ui.debugMode && (
              <div className="fixed bottom-4 right-4 bg-gray-900 text-white text-xs px-2 py-1 rounded">
                Debug Mode
              </div>
            )}
          </div>
        </Providers>
      </body>
    </html>
  );
}