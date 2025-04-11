/**
 * Utility functions for working with route parameters
 */

/**
 * Validates and extracts the id parameter from dynamic routes
 * @param params The params object from a dynamic route
 * @returns A validated id string
 * @throws Error if id is not a string
 */
export function getValidatedId(params: { id?: string | string[] } | undefined): string {
  if (!params) {
    throw new Error('No parameters provided');
  }
  
  if (!params.id) {
    throw new Error('No id parameter found');
  }
  
  // Handle both string and array cases (Next.js can provide either)
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  
  if (typeof id !== 'string' || id.trim() === '') {
    throw new Error('Invalid id parameter');
  }
  
  return id;
} 