import type { NextConfig } from "next";

// Multi-Zones: in production this app is mounted under /lab/brain on
// www.artjeck.com via a rewrite from the main site. Setting basePath makes
// every route + asset resolve under that prefix. Unset (local dev) → root.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || undefined;

const nextConfig: NextConfig = {
  basePath,
};

export default nextConfig;
