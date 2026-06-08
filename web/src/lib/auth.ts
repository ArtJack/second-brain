import { compare } from "bcryptjs";
import type { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Owner",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        const expectedUsername = process.env.OWNER_USERNAME?.trim() || "ArtJack";
        const passwordHash = process.env.OWNER_PASSWORD_HASH?.trim();
        const username = credentials?.username?.trim();
        const password = credentials?.password ?? "";

        if (!passwordHash || username !== expectedUsername) {
          return null;
        }

        const passwordOk = await compare(password, passwordHash);
        if (!passwordOk) {
          return null;
        }

        return {
          id: "owner",
          name: expectedUsername,
          role: "owner",
        };
      },
    }),
  ],
  callbacks: {
    jwt({ token, user }) {
      if (user) {
        token.role = user.role;
      }
      return token;
    },
    session({ session, token }) {
      if (session.user) {
        session.user.role = token.role === "owner" ? "owner" : undefined;
      }
      return session;
    },
  },
  session: {
    strategy: "jwt",
  },
  secret: process.env.AUTH_SECRET ?? process.env.NEXTAUTH_SECRET,
};
