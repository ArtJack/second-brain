// Multi-Zones: this app is served at /lab/brain on www.artjeck.com (set via
// NEXT_PUBLIC_BASE_PATH at build time) but at the root in local dev.
// Next's `basePath` auto-prefixes <Link>, router, and <Image>, but NOT raw
// fetch() calls or the next-auth client — so we thread the prefix through those
// manually with this helper.
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export function withBase(path: string): string {
  return `${BASE_PATH}${path}`;
}
