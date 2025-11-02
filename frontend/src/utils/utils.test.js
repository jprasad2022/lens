// Simple utility function tests
describe('Utility Functions', () => {
  it('should pass a basic test', () => {
    expect(1 + 1).toBe(2);
  });

  it('should handle string operations', () => {
    const text = 'MovieLens';
    expect(text.toLowerCase()).toBe('movielens');
    expect(text.length).toBe(9);
  });

  it('should handle array operations', () => {
    const movies = ['Movie 1', 'Movie 2', 'Movie 3'];
    expect(movies.length).toBe(3);
    expect(movies[0]).toBe('Movie 1');
  });
});