import { FiAlertCircle, FiRefreshCw } from 'react-icons/fi';

export default function ErrorMessage({ title, message, onRetry }) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md mx-auto">
      <div className="flex items-start">
        <FiAlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
        <div className="flex-1">
          <h3 className="text-lg font-medium text-red-900">{title}</h3>
          {message && (
            <p className="mt-1 text-sm text-red-700">{message}</p>
          )}
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-4 inline-flex items-center px-3 py-1.5 text-sm font-medium text-red-700 bg-red-100 hover:bg-red-200 rounded-md transition-colors"
            >
              <FiRefreshCw className="w-4 h-4 mr-1.5" />
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}