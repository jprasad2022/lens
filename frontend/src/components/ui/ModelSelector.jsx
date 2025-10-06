'use client';

import { useApp } from '@/contexts/AppContext';
import toast from 'react-hot-toast';

export default function ModelSelector() {
  const { selectedModel, availableModels, setSelectedModel } = useApp();

  const handleModelChange = async (e) => {
    const newModel = e.target.value;
    try {
      await setSelectedModel(newModel);
      toast.success(`Switched to ${newModel} model`);
    } catch (error) {
      toast.error('Failed to switch model');
      console.error('Model switch error:', error);
    }
  };

  return (
    <div>
      <label htmlFor="model-select" className="label">
        Recommendation Model
      </label>
      <select
        id="model-select"
        value={selectedModel}
        onChange={handleModelChange}
        className="input"
      >
        {availableModels.map((model) => (
          <option key={model} value={model}>
            {model.charAt(0).toUpperCase() + model.slice(1)}
          </option>
        ))}
      </select>
    </div>
  );
}