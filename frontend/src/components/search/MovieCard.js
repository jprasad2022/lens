
import Image from 'next/image';
import Rating from './Rating';

export default function MovieCard({ movie }) {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="relative h-64">
        <Image
          src={movie.poster_url || '/placeholder.png'}
          alt={movie.title}
          layout="fill"
          objectFit="cover"
        />
      </div>
      <div className="p-4">
        <h3 className="font-bold text-lg mb-2">{movie.title}</h3>
        <Rating movieId={movie.id} />
      </div>
    </div>
  );
}
