"use client";

import { SessionProvider } from "next-auth/react";

import { BASE_PATH } from "@/lib/base-path";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // Tell the next-auth client where its API lives under the base path so
  // signIn/signOut hit /lab/brain/api/auth in production (root in dev).
  return <SessionProvider basePath={`${BASE_PATH}/api/auth`}>{children}</SessionProvider>;
}
