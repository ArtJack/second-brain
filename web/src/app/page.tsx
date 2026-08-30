import { AuthProvider } from "@/components/auth-provider";
import { BrainApp } from "@/components/brain-app";

export default function Home() {
  // Server component: only it can see the env. The owner login control is
  // pointless UI on a deployment that deliberately carries no owner
  // credentials, so the public demo hides it entirely.
  const ownerLoginEnabled = Boolean(process.env.OWNER_PASSWORD_HASH?.trim());
  return (
    <AuthProvider>
      <BrainApp ownerLoginEnabled={ownerLoginEnabled} />
    </AuthProvider>
  );
}
