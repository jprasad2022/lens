'use client';

export default function UserInput({ value, onChange, placeholder, label }) {
  return (
    <div>
      <label className="label">
        {label}
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="input"
      />
    </div>
  );
}