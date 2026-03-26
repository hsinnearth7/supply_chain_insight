import { useEffect, useState, useRef } from 'react';
import { auth } from '../api/client';
import LoadingSpinner from './LoadingSpinner';

interface ChartImageProps {
  src: string;
  alt: string;
  className?: string;
}

export default function ChartImage({ src, alt, className = '' }: ChartImageProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const prevUrlRef = useRef<string | null>(null);

  useEffect(() => {
    let active = true;

    // Revoke previous URL
    if (prevUrlRef.current) {
      URL.revokeObjectURL(prevUrlRef.current);
      prevUrlRef.current = null;
    }

    async function loadImage() {
      setLoading(true);
      setError(false);
      setObjectUrl(null);

      try {
        const res = await fetch(src, { headers: auth.buildHeaders() });
        if (!res.ok) {
          throw new Error(`Chart request failed: ${res.status}`);
        }
        const blob = await res.blob();
        const nextObjectUrl = URL.createObjectURL(blob);
        if (!active) {
          URL.revokeObjectURL(nextObjectUrl);
          return;
        }
        setObjectUrl(nextObjectUrl);
        prevUrlRef.current = nextObjectUrl;
      } catch {
        if (active) {
          setError(true);
          setLoading(false);
        }
      }
    }

    void loadImage();

    return () => {
      active = false;
      if (prevUrlRef.current) {
        URL.revokeObjectURL(prevUrlRef.current);
        prevUrlRef.current = null;
      }
    };
  }, [src]);

  return (
    <div className={`relative bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden ${className}`}>
      {loading && !error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <LoadingSpinner size="sm" />
        </div>
      )}
      {error ? (
        <div className="flex items-center justify-center h-48 text-ci-gray text-sm">
          Chart not available
        </div>
      ) : objectUrl ? (
        <img
          src={objectUrl}
          alt={alt}
          className={`w-full h-auto transition-opacity ${loading ? 'opacity-0' : 'opacity-100'}`}
          onLoad={() => setLoading(false)}
          onError={() => { setError(true); setLoading(false); }}
          loading="lazy"
        />
      ) : (
        <div className="flex items-center justify-center h-48">
          <LoadingSpinner size="sm" />
        </div>
      )}
    </div>
  );
}
