import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import HomePage from './page';

describe('HomePage', () => {
  it('renders without crashing', () => {
    render(<HomePage />);
    expect(screen.getByText(/Movie Recommendation System/i)).toBeInTheDocument();
  });

  it('renders user selection section', () => {
    render(<HomePage />);
    expect(screen.getByText(/Select a User/i)).toBeInTheDocument();
  });

  it('renders demo user buttons', () => {
    render(<HomePage />);
    expect(screen.getByText(/Alex/i)).toBeInTheDocument();
    expect(screen.getByText(/Sam/i)).toBeInTheDocument();
    expect(screen.getByText(/Morgan/i)).toBeInTheDocument();
    expect(screen.getByText(/Jordan/i)).toBeInTheDocument();
  });
});