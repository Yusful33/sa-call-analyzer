/**
 * Simple client-side cache with stale-while-revalidate pattern.
 * Uses localStorage for persistence across page reloads.
 */

type CacheEntry<T> = {
  data: T;
  timestamp: number;
  /** Max age in milliseconds before data is considered stale */
  maxAge: number;
};

const CACHE_PREFIX = "stillness_cache_";

/**
 * Get cached data if available and not expired.
 * Returns { data, isStale } where isStale indicates the data should be refreshed in background.
 */
export function getCached<T>(key: string): { data: T; isStale: boolean } | null {
  if (typeof window === "undefined") return null;
  
  try {
    const raw = localStorage.getItem(CACHE_PREFIX + key);
    if (!raw) return null;
    
    const entry: CacheEntry<T> = JSON.parse(raw);
    const age = Date.now() - entry.timestamp;
    
    // If data is older than 2x maxAge, consider it too stale to use
    if (age > entry.maxAge * 2) {
      localStorage.removeItem(CACHE_PREFIX + key);
      return null;
    }
    
    return {
      data: entry.data,
      isStale: age > entry.maxAge,
    };
  } catch {
    return null;
  }
}

/**
 * Store data in cache with a max age.
 */
export function setCache<T>(key: string, data: T, maxAgeMs: number): void {
  if (typeof window === "undefined") return;
  
  try {
    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      maxAge: maxAgeMs,
    };
    localStorage.setItem(CACHE_PREFIX + key, JSON.stringify(entry));
  } catch {
    // localStorage might be full or disabled
  }
}

/**
 * Clear a specific cache entry.
 */
export function clearCache(key: string): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(CACHE_PREFIX + key);
}

/**
 * Clear all Stillness cache entries.
 */
export function clearAllCache(): void {
  if (typeof window === "undefined") return;
  
  const keysToRemove: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key?.startsWith(CACHE_PREFIX)) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((k) => localStorage.removeItem(k));
}
